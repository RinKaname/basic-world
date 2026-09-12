import string

class BasicWordTokenizer:
    def __init__(self):
        self.word2id = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
        self.id2word = {0: "[PAD]", 1: "[UNK]", 2: "[EOS]"}
        self.vocab_size = 3

    def build_vocab(self, filepath):
        with open(filepath, 'r') as f:
            text = f.read()

        # Isolate the period so it can be parsed as its own token [EOS]
        text = text.replace(".", " [EOS] ")

        # Remove all other punctuation
        punct_to_remove = string.punctuation.replace(".", "")
        translator = str.maketrans('', '', punct_to_remove)
        clean_text = text.translate(translator)

        # Lowercase everything except the special tags
        clean_text = clean_text.replace("[EOS]", "<TEMP_EOS>")
        clean_text = clean_text.lower()
        clean_text = clean_text.replace("<temp_eos>", "[EOS]")

        words = clean_text.split()

        # Remove [EOS] from the raw words list since it's hardcoded at ID 2
        words = [w for w in words if w not in ["[eos]", "[EOS]", "eos"]]

        unique_words = sorted(list(set(words)))

        for word in unique_words:
            if word not in self.word2id:
                self.word2id[word] = self.vocab_size
                self.id2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text):
        # Isolate the period
        text = text.replace(".", " [EOS] ")

        punct_to_remove = string.punctuation.replace(".", "")
        translator = str.maketrans('', '', punct_to_remove)
        clean_text = text.translate(translator)

        # Lowercase everything except the special tags
        clean_text = clean_text.replace("[EOS]", "<TEMP_EOS>")
        clean_text = clean_text.lower()
        clean_text = clean_text.replace("<temp_eos>", "[EOS]")

        words = clean_text.split()

        # [EOS] is at ID 2
        return [2 if word in ["[eos]", "[EOS]", "eos"] else self.word2id.get(word, self.word2id["[UNK]"]) for word in words]

    def decode(self, token_ids):
        decoded_words = [self.id2word.get(tid, "[UNK]") for tid in token_ids if tid != 0]
        # Clean up the output string formatting
        out_str = " ".join(decoded_words)
        # Handle capitalized versions from id2word (which we inserted as [EOS])
        # Sometimes decoding lowercases or preserves raw string based on how it was mapped.
        out_str = out_str.replace(" [EOS]", ".").replace("[EOS]", ".")
        out_str = out_str.replace(" [eos]", ".").replace("[eos]", ".")
        out_str = out_str.replace(" eos", ".") # In case it got entirely lowercased during decoding without brackets

        # Make sure the period doesn't have a space before it
        out_str = out_str.replace(" .", ".")

        # Capitalize the first letter
        if out_str:
            out_str = out_str[0].upper() + out_str[1:]

        return out_str.strip()

if __name__ == "__main__":
    tok = BasicWordTokenizer()
    tok.build_vocab("text/world_word.txt")
    print(f"Vocab Size: {tok.vocab_size}")

    test_sentence = "I hunt deer."
    encoded = tok.encode(test_sentence)
    decoded = tok.decode(encoded)

    print(f"Original: {test_sentence}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
