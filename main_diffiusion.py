import os
import glob
import time
import math
import random
import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & TOKENIZER
# -----------------------------------------------------------------------------

SPECIAL_TOKEN_MAP = {
    "<PAD>":    0xC0,  # 192
    "<EOS>":    0xC1,  # 193
    "<DELETE>": 0xF5,  # 245
    "<EXPAND>": 0xF6,  # 246
    "<MASK>":   0xF7,  # 247
}
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKEN_MAP.items()}
VOCAB_SIZE = 256

class ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        valid_bytes = bytearray()
        for t in tokens:
            if t not in ID_TO_SPECIAL:
                valid_bytes.append(t)
        return valid_bytes.decode("utf-8", errors="replace")
def process_large_file(model, text, device, window_size=512, overlap=256):
    """
    Processes a long string by sliding a window with overlap.
    Merges outputs using a linear fade (sigmoid-like) weight to prevent boundary seams.
    """
    tokenizer = ByteTokenizer()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    N = len(data)
    
    # We will accumulate weighted logits for each position
    # Shape: (N, Vocab_Size)
    # Note: For huge files, keep this on CPU to save VRAM/RAM
    all_logits = torch.zeros(N, 256, device="cpu")
    weights = torch.zeros(N, device="cpu")
    
    # Create a "fade mask" (trapezoid shape) for the window
    # 0 -> 1 -> 1 -> 0
    # This ensures we trust the center of the window most
    ramp = torch.linspace(0, 1, overlap)
    window_mask = torch.ones(window_size)
    window_mask[:overlap] = ramp
    window_mask[-overlap:] = 1 - ramp
    
    model.eval()
    step = window_size - overlap
    
    with torch.no_grad():
        for start in range(0, N, step):
            end = min(start + window_size, N)
            
            # Prepare chunk
            chunk = data[start:end]
            pad_len = window_size - len(chunk)
            
            # If chunk is too short (end of file), pad it
            if pad_len > 0:
                chunk = torch.cat([chunk, torch.full((pad_len,), 0xC0, dtype=torch.long)])
            
            # Run Model
            inp = chunk.unsqueeze(0).to(device) # (1, 512)
            
            # Get logits
            # For diffusion, you might run this repeatedly. 
            # Here we assume a single pass corrector for simplicity.
            logits = model(inp)[0].cpu() # (512, 256)
            
            # Remove padding from output
            valid_len = end - start
            logits = logits[:valid_len]
            mask = window_mask[:valid_len]
            
            # Accumulate
            # We add the weighted logits to the global buffer
            # (broadcasting mask to shape (Len, 1))
            all_logits[start:end] += logits * mask.unsqueeze(1)
            weights[start:end] += mask
            
    # Normalize and Decode
    # Avoid division by zero
    weights[weights == 0] = 1.0 
    final_logits = all_logits / weights.unsqueeze(1)
    
    pred_ids = final_logits.argmax(dim=-1).tolist()
    return tokenizer.decode(pred_ids)
# -----------------------------------------------------------------------------
# 2. MEMORY-SAFE DATA PIPELINE (Streaming)
# -----------------------------------------------------------------------------

class VocabAnalyzer:
    def __init__(self):
        self.counts = np.zeros(256, dtype=np.int64)
        self.total = 0
        self.counts += 1 

    def update(self, text_bytes):
        unique, counts = np.unique(list(text_bytes), return_counts=True)
        for u, c in zip(unique, counts):
            if u < 256 and u not in ID_TO_SPECIAL:
                self.counts[u] += c
        self.total += len(text_bytes)

    def get_probs(self):
        probs = self.counts.astype(np.float32)
        for sp_id in SPECIAL_TOKEN_MAP.values():
            if sp_id < 256: probs[sp_id] = 0.0
        return probs / probs.sum()

class CorruptionEngine:
    def __init__(self, vocab_probs=None):
        self.vocab_probs = vocab_probs if vocab_probs is not None else np.ones(256)/256

    def get_noise_token(self):
        return np.random.choice(256, p=self.vocab_probs)

    def process_segment(self, clean_bytes):
        src = []
        tgt = []
        i = 0
        n = len(clean_bytes)
        
        while i < n:
            r = random.random()
            # Op 1: Insertion
            if r < 0.05: 
                src.append(self.get_noise_token())
                tgt.append(SPECIAL_TOKEN_MAP["<DELETE>"])
                continue
            # Op 2: Expansion
            elif r < 0.10:
                span_len = random.randint(1, 8)
                span_len = min(span_len, n - i)
                src.append(SPECIAL_TOKEN_MAP["<EXPAND>"])
                src.extend([SPECIAL_TOKEN_MAP["<PAD>"]] * (span_len - 1))
                tgt.extend([SPECIAL_TOKEN_MAP["<MASK>"]] * span_len)
                i += span_len
                continue
            # Op 3: Masking
            elif r < 0.25:
                src.append(SPECIAL_TOKEN_MAP["<MASK>"])
                tgt.append(clean_bytes[i])
                i += 1
            # Identity
            else:
                src.append(clean_bytes[i])
                tgt.append(clean_bytes[i])
                i += 1
        return src, tgt

def correction_data_generator(filename_pattern, batch_size, max_seq_len, device="cpu"):
    files = sorted(glob.glob(filename_pattern))
    if not files:
        print("No files found.")
        files = []

    # 1. Quick Histogram (Read only 1st MB)
    print("Building vocabulary histogram...")
    analyzer = VocabAnalyzer()
    sample_bytes = 0
    for fpath in files:
        if sample_bytes > 1024*1024: break
        try:
            with open(fpath, 'rb') as f:
                analyzer.update(f.read(50000))
                sample_bytes += 50000
        except: continue
    vocab_probs = analyzer.get_probs()

    tokenizer = ByteTokenizer()
    engine = CorruptionEngine(vocab_probs)
    file_cycle = itertools.cycle(files) if files else None
    
    # 2. Streaming Loop
    # We maintain a buffer to assemble batches across file chunks
    buffer_src = []
    buffer_tgt = []
    
    # Read files in small chunks (e.g. 1MB) to prevent RAM explosion
    CHUNK_SIZE = 1024 * 1024 
    
    while True:
        if not files:
            # Dummy data fallback
            clean_bytes = tokenizer.encode("Hello world, this is a test. " * 50)
            s, t = engine.process_segment(clean_bytes)
            buffer_src.extend(s)
            buffer_tgt.extend(t)
        else:
            curr_file = next(file_cycle)
            try:
                with open(curr_file, 'r', encoding='utf-8', errors='ignore') as f:
                    while True:
                        # READ CHUNK INSTEAD OF WHOLE FILE
                        text_chunk = f.read(CHUNK_SIZE)
                        if not text_chunk: break
                        
                        clean_bytes = tokenizer.encode(text_chunk)
                        s, t = engine.process_segment(clean_bytes)
                        buffer_src.extend(s)
                        buffer_tgt.extend(t)
                        
                        # Yield batches as soon as we have enough data
                        while len(buffer_src) >= batch_size * max_seq_len:
                            # Extract one batch worth of tokens
                            # Note: We cut strictly by length here. 
                            # Ideally we'd respect sentences, but for char-level diffusion 
                            # strict chunking is acceptable training noise.
                            batch_src_list = []
                            batch_tgt_list = []
                            
                            # Slice out 'batch_size' sequences
                            total_needed = batch_size * max_seq_len
                            
                            # Grab raw tokens
                            raw_s = buffer_src[:total_needed]
                            raw_t = buffer_tgt[:total_needed]
                            
                            # Clear from buffer
                            buffer_src = buffer_src[total_needed:]
                            buffer_tgt = buffer_tgt[total_needed:]
                            
                            # Reshape into (B, T)
                            for i in range(0, total_needed, max_seq_len):
                                batch_src_list.append(raw_s[i : i+max_seq_len])
                                batch_tgt_list.append(raw_t[i : i+max_seq_len])
                                
                            _inputs = torch.tensor(batch_src_list, dtype=torch.long)
                            _targets = torch.tensor(batch_tgt_list, dtype=torch.long)
                            
                            use_non_blocking = (device == "cuda")
                            yield (
                                _inputs.to(device=device, non_blocking=use_non_blocking),
                                _targets.to(device=device, non_blocking=use_non_blocking)
                            )
            except Exception as e:
                print(f"Error reading file {curr_file}: {e}")
                continue

# -----------------------------------------------------------------------------
# 3. OPTIMIZER & MATH
# -----------------------------------------------------------------------------

coeffs_list = [
    (8.2872, -23.5959, 17.3004), (4.1071, -2.9478, 0.5448), 
    (3.9487, -2.9089, 0.5518), (3.3184, -2.4885, 0.5100), 
    (2.3007, -1.6689, 0.4188), (1.8913, -1.2680, 0.3768), 
    (1.8750, -1.2500, 0.3750), (1.875, -1.25, 0.375)
]
coeffs_list = [(a/1.01, b/1.01**3, c/1.01**5) for (a,b,c) in coeffs_list[:-1]] + [coeffs_list[-1]]

def polar_express(G: torch.Tensor, steps: int) -> torch.Tensor:
    assert G.ndim == 2
    dtype = torch.bfloat16 if (G.device.type == 'cpu' and torch.cuda.is_bf16_supported()) else torch.float32
    X = G.to(dtype=dtype)
    if G.size(0) > G.size(1): X = X.T
    X = X / (X.norm() + 1e-7)
    hs = coeffs_list[:steps] + list(itertools.repeat(coeffs_list[-1], steps - len(coeffs_list)))
    for a, b, c in hs:
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(dtype=G.dtype)

class NorMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.9, weight_decay=0.01, 
                 ns_steps=5, adam_lr=1e-3, adam_betas=(0.9, 0.95), epsilon=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, 
                        ns_steps=ns_steps, epsilon=epsilon,
                        adam_lr=adam_lr, adam_betas=adam_betas)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure: loss = closure()
        for group in self.param_groups:
            lr, beta_m, wd = group['lr'], group['momentum'], group['weight_decay']
            adam_lr, (beta1, beta2) = group['adam_lr'], group['adam_betas']
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p)
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if p.ndim >= 2:
                        state['vt'] = torch.zeros(p.size(0), 1, device=p.device, dtype=p.dtype)
                state['step'] += 1
                if p.ndim >= 2:
                    p_flat = p.view(p.size(0), -1)
                    g_flat = grad.view(p.size(0), -1)
                    buf = state['momentum'].view(p.size(0), -1)
                    buf.mul_(beta_m).add_(g_flat, alpha=1 - beta_m)
                    Ot = polar_express(buf, steps=group['ns_steps'])
                    Ot_sq_mean = Ot.square().mean(dim=1, keepdim=True)
                    vt = state['vt']
                    vt.mul_(beta1).add_(Ot_sq_mean, alpha=1 - beta1)
                    O_hat = Ot / (vt + group['epsilon'])
                    scale = 0.2 * lr * (p_flat.shape[0]*p_flat.shape[1])**0.5 / (O_hat.norm() + 1e-9)
                    p_flat.mul_(1 - lr * wd)
                    p_flat.add_(O_hat, alpha=-scale)
                else:
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    p.mul_(1 - adam_lr * wd)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.sqrt().add_(group['epsilon'])
                    p.addcdiv_(exp_avg, denom, value=-adam_lr)
        return loss

# -----------------------------------------------------------------------------
# 4. MODEL (Optimized for ~10M Params)
# -----------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Cache as (1, 1, T, D) for broadcasting against (B, H, T, D)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x):
        # x: (B, H, T, D)
        seq_len = x.shape[2]
        return self.cos_cached[:, :, :seq_len, :], self.sin_cached[:, :, :seq_len, :]

def apply_rope(q, k, cos, sin):
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out

class BidirectionalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. SETUP DIMENSIONS
        # Standard: head_dim = 48 (for 384 model / 8 heads)
        # Paired Head Strategy: Squeeze QK (speed), Expand V (capacity)
        
        self.n_head = config.n_head
        self.embd = config.n_embd
        
        
        self.qk_head_dim = config.qk_head_dim 
        
        self.v_head_dim = config.v_head_dim
        
        # 2. DECOUPLED PROJECTIONS
        # We can no longer use a single Linear(dim, 3*dim)
        
        # Query & Key: Output size = n_head * 32
        self.q_proj = nn.Linear(self.embd, self.n_head * self.qk_head_dim, bias=False)
        self.k_proj = nn.Linear(self.embd, self.n_head * self.qk_head_dim, bias=False)
        
        # Value: Output size = n_head * 64
        self.v_proj = nn.Linear(self.embd, self.n_head * self.v_head_dim, bias=False)
        
        # Output Projection: Maps n_head * 64 back to model dim (384)
        self.c_proj = nn.Linear(self.n_head * self.v_head_dim, self.embd, bias=False)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.size()
        
        # 1. Project Q, K (Small)
        q = self.q_proj(x).view(B, T, self.n_head, self.qk_head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.qk_head_dim).transpose(1, 2)
        
        # 2. Project V (Large)
        v = self.v_proj(x).view(B, T, self.n_head, self.v_head_dim).transpose(1, 2)
        
        # 3. Apply RoPE (Only to Q and K)
        # Note: RoPE implementation needs to accept the smaller qk_head_dim now
        # You must ensure the RoPE cache logic handles 'qk_head_dim' (32), not 'v_head_dim' (64).
        # Since our fixed RoPE creates freq based on input size, it should work if slice is correct.
        q, k = apply_rope(q, k, rope_cos, rope_sin)
        
        # 4. Attention
        # PyTorch F.sdpa supports different V dimensions natively!
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        
        # 5. Reassemble
        # y shape: (B, Heads, T, v_head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.v_head_dim)
        
        return self.c_proj(y)

# class BidirectionalAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         assert config.n_embd % config.n_head == 0
#         self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
#         self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
#         self.n_head = config.n_head
#         self.head_dim = config.n_embd // config.n_head

#     def forward(self, x, rope_cos, rope_sin):
#         B, T, C = x.size()
#         qkv = self.c_attn(x).chunk(3, dim=2)
#         q, k, v = [t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in qkv]
#         q, k = apply_rope(q, k, rope_cos, rope_sin)
#         y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
#         y = y.transpose(1, 2).contiguous().view(B, T, C)
#         return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln_1(x), rope_cos, rope_sin)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 4096
    vocab_size: int = 256
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 256
    qk_head_dim: int = 32 # QK Head Dim: Keep small (e.g., 32) for fast attention score calc
    v_head_dim: int = 64 # V Head Dim: Keep large (e.g., 64) for moving more information
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        
        # REMOVED: BigramEmbedding (Saved ~25M params)
        # To restore: Uncomment and add " + self.bigram_emb(idx)" in forward
        # self.bigram_emb = nn.Embedding(config.vocab_size**2, config.n_embd)

        # self.rope = RotaryEmbedding(config.n_embd // config.n_head, config.block_size)
        # To matching the qk_head_dim=32 we set above):
        self.rope = RotaryEmbedding(config.qk_head_dim, config.block_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.token_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        
        # Standard embedding
        x = self.token_emb(idx) 
        
        # Dummy RoPE call to get cos/sin for current seq_len
        # (Pass shape B,H,T,D -> we just need T)
        dummy_x = x.unsqueeze(1) 
        cos, sin = self.rope(dummy_x)
        
        skips = []
        half_layers = len(self.blocks) // 2
        
        for i, block in enumerate(self.blocks):
            if i < half_layers:
                x = block(x, cos, sin)
                skips.append(x)
            else:
                skip_val = skips.pop()
                x = x + skip_val 
                x = block(x, cos, sin)
                
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits



# -----------------------------------------------------------------------------
# 5. MAIN (With Resume & Safe Paths)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Configuration ---
    # Determine the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = "/media/gall/SSD/TeX/"#os.path.join(SCRIPT_DIR, "data")       # Look for data here
    CKPT_DIR = os.path.join(SCRIPT_DIR, "checkpoints") # Save checkpoints here
    os.makedirs(CKPT_DIR, exist_ok=True)

    TRAIN_PATTERN = os.path.join(DATA_DIR, "*.tex")#"*.txt")
    BATCH_SIZE = 32
    SEQ_LEN = 512
    MAX_STEPS = 50000
    SAVE_EVERY = 500
    
    DEVICE = "cpu"
    if torch.cuda.is_available(): DEVICE = "cuda"
    elif torch.backends.mps.is_available(): DEVICE = "mps"
    
    print(f"--- Training Config ---")
    print(f"Device:      {DEVICE}")
    print(f"Data Path:   {TRAIN_PATTERN}")
    print(f"Ckpt Path:   {CKPT_DIR}")
    print(f"Arch:        ~10M Params, Bidirectional, RoPE, U-Net, Pair head attention")
    
    # --- Init Model & Optimizer ---
    config = GPTConfig(block_size=SEQ_LEN)
    model = GPT(config).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params/1e6:.2f}M")

    optimizer = NorMuon(model.parameters(), lr=0.05, adam_lr=0.001, ns_steps=5)
    
    # --- Checkpoint Loading Logic ---
    start_step = 0
    # Find all checkpoint_*.pt files
    ckpt_files = glob.glob(os.path.join(CKPT_DIR, "checkpoint_*.pt"))
    if ckpt_files:
        try:
            # Find latest checkpoint
            latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            print(f"Attempting to resume from: {latest_ckpt}")

            # 1. OPTION A: Secure Load (PyTorch 2.6+)
            # We must tell PyTorch that GPTConfig is safe to unpickle
            try:
                import torch.serialization
                torch.serialization.add_safe_globals([GPTConfig])
                checkpoint = torch.load(latest_ckpt, map_location=DEVICE, weights_only=True)
            except (AttributeError, RuntimeError, ImportError):
                # Fallback for older PyTorch or if whitelist fails
                print("Warning: Secure load failed, falling back to weights_only=False")
                checkpoint = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)

            # 2. Load Model Weights (Strict=True ensures architecture matches)
            model.load_state_dict(checkpoint['model_state'])
            print("Model weights loaded.")

            # 3. Clever Optimizer Load
            # We try to load the optimizer, but if it fails (e.g. slight mismatch), 
            # we skip it and restart the optimizer logic.
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
                start_step = checkpoint['step'] + 1
                print(f"Optimizer state loaded. Resuming at step {start_step}")
            except Exception as e:
                print(f"Warning: Could not load optimizer state ({e}).")
                print("Reseting optimizer and continuing training from current weights.")
                # start_step remains 0 (or you can set it to checkpoint['step'] if you just want to track progress)
                # Usually if we reset optimizer, we treat it as a 'finetune' start, but keeping step count is often useful for logs.
                start_step = checkpoint['step'] + 1

        except Exception as e:
            print(f"CRITICAL: Failed to load checkpoint {latest_ckpt}: {e}")
            print("Starting training from scratch.")

    # --- Data Loader ---
    train_loader = correction_data_generator(TRAIN_PATTERN, BATCH_SIZE, SEQ_LEN, DEVICE)
    
    # --- Training Loop ---
    model.train()
    print("Starting training...")
    
    t0 = time.time()
    
    # Adjust range to start from resumed step
    for step in range(start_step, MAX_STEPS):
        inputs, targets = next(train_loader)
        
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.view(-1, VOCAB_SIZE), 
            targets.view(-1), 
            ignore_index=SPECIAL_TOKEN_MAP["<PAD>"]
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        if step % 10 == 0:
            t1 = time.time()
            dt = (t1 - t0) * 1000
            t0 = t1 # Reset timer
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | Time: {dt:.2f}ms")
            
        if step % SAVE_EVERY == 0 and step > 0:
            print("\n--- INFERENCE CHECK ---")
            with torch.no_grad():
                inp_seq = inputs[0].tolist()
                tar_seq = targets[0].tolist()
                pred_seq = logits[0].argmax(dim=-1).tolist()
                tok = ByteTokenizer()
                print(f"INPUT:  {tok.decode(inp_seq)[:80]}...")
                print(f"TARGET: {tok.decode(tar_seq)[:80]}...")
                print(f"MODEL:  {tok.decode(pred_seq)[:80]}...")
            print("-----------------------\n")
            
            # Save Checkpoint
            ckpt_path = os.path.join(CKPT_DIR, f"checkpoint_{step}.pt")
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'config': config, # Good practice to save config
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")