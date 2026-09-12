import string

class BasicWordTokenizer:
    def __init__(self):
        self.word2id = {"[PAD]": 0, "[UNK]": 1}
        self.id2word = {0: "[PAD]", 1: "[UNK]"}
        self.vocab_size = 2

    def build_vocab(self, filepath):
        with open(filepath, 'r') as f:
            text = f.read()

        translator = str.maketrans('', '', string.punctuation)
        clean_text = text.translate(translator).lower()

        words = clean_text.split()
        unique_words = sorted(list(set(words)))

        for word in unique_words:
            if word not in self.word2id:
                self.word2id[word] = self.vocab_size
                self.id2word[self.vocab_size] = word
                self.vocab_size += 1

    def encode(self, text):
        translator = str.maketrans('', '', string.punctuation)
        clean_text = text.translate(translator).lower()
        words = clean_text.split()

        return [self.word2id.get(word, self.word2id["[UNK]"]) for word in words]

    def decode(self, token_ids):
        return " ".join([self.id2word.get(tid, "[UNK]") for tid in token_ids if tid != 0]).capitalize() + "."

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
