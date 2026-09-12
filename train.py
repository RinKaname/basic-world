import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import string

from tokenizer import BasicWordTokenizer
from model import AmadeusRNN

class BasicWorldDataset(Dataset):
    def __init__(self, filepath, tokenizer, seq_len=8):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        with open(filepath, "r") as f:
            lines = f.readlines()

        # Flatten all sentences into one continuous stream of words
        full_text = " ".join(lines)
        translator = str.maketrans('', '', string.punctuation)
        clean_text = full_text.translate(translator).lower()

        self.words = clean_text.split()
        self.tokens = [tokenizer.word2id.get(w, tokenizer.word2id["[UNK]"]) for w in self.words]

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.seq_len + 1]
        return torch.tensor(chunk, dtype=torch.long)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧠 AMADEUS-RNN — BASIC WORLD EDITION")
    print("=" * 60)

    # Check Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using Device: {DEVICE.upper()}")

    # Config
    DATA_PATH = "text/world_word.txt"
    H = 64            # Tiny hidden dimension
    B = 16            # Batch size
    T = 6             # Sequence length (short basic sentences)
    LR = 1e-3         # Faster learning rate for tiny model
    EPOCHS = 100      # Since data is tiny, we need more epochs to memorize the grammar

    # 1. Setup Tokenizer
    tokenizer = BasicWordTokenizer()
    tokenizer.build_vocab(DATA_PATH)
    V = tokenizer.vocab_size
    print(f"📖 Vocab Size: {V} unique words")

    # 2. Setup Data
    dataset = BasicWorldDataset(DATA_PATH, tokenizer, seq_len=T)
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, drop_last=True)
    print(f"📚 Training Sequences: {len(dataset)}")

    # 3. Setup Model
    model = AmadeusRNN(vocab_size=V, hidden_dim=H, pad_token_id=0).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Total Parameters: {total_params:,} (Extremely Lightweight!)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 4. Training Loop
    print("\n🚀 Starting Training...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for batch in dataloader:
            batch = batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(batch)                  # (B, T, V)
            targets = batch[:, 1:]                 # (B, T)

            loss = F.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if epoch % 10 == 0 or epoch == 1:
            print(f"📊 Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f}")

            # Quick generation test
            prompt = "I Walk"
            encoded = tokenizer.encode(prompt)
            input_tensor = torch.tensor(encoded, dtype=torch.long, device=DEVICE).unsqueeze(0)

            with torch.no_grad():
                gen_tokens = model.generate(input_tensor, max_new_tokens=4, temperature=0.7)

            output_text = tokenizer.decode(gen_tokens[0].cpu().numpy().tolist())
            print(f"  🗣️  {prompt} -> {output_text}")

    print("\n✅ Training Complete!")

    # Save Model
    torch.save(model.state_dict(), "basic_world_model.pth")
    print("💾 Model saved to basic_world_model.pth")
