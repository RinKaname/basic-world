import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AmadeusRNN(nn.Module):
    """
    Parallel Linear RNN with data-dependent gating and associative scan.
    Architecture identical to PureMath_Amadeus_RNN, adapted for basic-world.
    """
    def __init__(self, vocab_size, hidden_dim=64, pad_token_id=0):
        super().__init__()
        self.V = vocab_size
        self.H = hidden_dim
        self.pad_id = pad_token_id

        # 1. Embedding (Spatial Bootstrap)
        self.W_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        nn.init.normal_(self.W_emb.weight, std=0.02)

        # 2. Recurrent matrix (Karl Working Memory)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 3. Data-dependent lambda gate (Mamba-style)
        self.W_lambda = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # 4. Decoder
        self.W_out = nn.Linear(hidden_dim, vocab_size, bias=False)

    def _associative_scan(self, lambda_t, u_t):
        """
        Parallel prefix scan: h_t = lambda_t * h_{t-1} + u_t
        O(T log T) work, O(log T) depth — GPU friendly.
        """
        B, T, H = lambda_t.shape

        mult = lambda_t
        adder = u_t
        steps = int(math.ceil(math.log2(T)))

        for i in range(steps):
            shift = 2 ** i

            mult_shifted = F.pad(mult[:, :-shift, :], (0, 0, shift, 0))
            adder_shifted = F.pad(adder[:, :-shift, :], (0, 0, shift, 0))

            mask = (torch.arange(T, device=lambda_t.device) >= shift).view(1, -1, 1)

            old_mult = mult
            mult = torch.where(mask, old_mult * mult_shifted, old_mult)
            adder = torch.where(mask, adder + old_mult * adder_shifted, adder)

        return adder

    def forward(self, X_raw):
        """Forward pass. Returns logits of shape (B, T-1, V)."""
        # Embedding + L2 normalize
        X_pos = self.W_emb(X_raw)
        X_pos_norm = F.normalize(X_pos, p=2, dim=-1)

        # Shift: predict token t+1 from tokens 0..t
        H_past = X_pos_norm[:, :-1, :]

        # Data-dependent gating
        lambda_t = torch.sigmoid(self.W_lambda(H_past))
        u_t = self.W_h(H_past)

        # Parallel associative scan
        H_pred = self._associative_scan(lambda_t, u_t)

        # Decode
        logits = self.W_out(H_pred)
        return logits

    @torch.no_grad()
    def generate(self, X_raw, max_new_tokens=10, temperature=1.0, top_k=None):
        """Autoregressive generation using incremental hidden state update."""
        self.eval()
        generated = X_raw.clone()

        # Encode the prompt with full scan
        X_pos = self.W_emb(generated)
        X_pos_norm = F.normalize(X_pos, p=2, dim=-1)

        lambda_t = torch.sigmoid(self.W_lambda(X_pos_norm))
        u_t = self.W_h(X_pos_norm)

        h_seq = self._associative_scan(lambda_t, u_t)
        h = h_seq[:, -1, :]  # Last hidden state

        for _ in range(max_new_tokens):
            logits = self.W_out(h) / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

            # Incremental state update: O(H²) per token
            x_next = self.W_emb(next_token)
            x_next_norm = F.normalize(x_next, p=2, dim=-1).squeeze(1)

            lam_next = torch.sigmoid(self.W_lambda(x_next_norm))
            u_next = self.W_h(x_next_norm)

            h = lam_next * h + u_next

        self.train()
        return generated
