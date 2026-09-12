import string

def calculate_tokens(filepath):
    with open(filepath, 'r') as f:
        text = f.read()

    # Remove punctuation (like periods) and convert to lowercase for accurate unique counting
    # Note: If we want "Apple" and "apple" to be the same token, we lowercase.
    # For a simple game, lowercasing everything makes sense.
    translator = str.maketrans('', '', string.punctuation)
    clean_text = text.translate(translator).lower()

    words = clean_text.split()

    total_words = len(words)
    unique_tokens = set(words)
    token_count = len(unique_tokens)

    print(f"Total Sentences: {text.count(chr(10))}")
    print(f"Total Words (Tokens Used): {total_words}")
    print(f"Unique Vocabulary Size (Custom Tokens): {token_count}")
    print("---")
    print(f"Unique Tokens: {sorted(list(unique_tokens))}")

if __name__ == '__main__':
    calculate_tokens('world_word.txt')
