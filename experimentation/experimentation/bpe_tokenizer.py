import collections
import json

class BpeTokenizer:
    def __init__(self, corpus=None, num_merges=15):
        self.corpus = corpus if corpus is not None else []
        self.num_merges = num_merges
        self.vocab = []
        self.merges = {}
        self.token_to_id = {}
        self.id_to_token = {}
        self.end_of_word = "</w>"

        if self.corpus:
            self.train(self.corpus)

    def _initialize_vocabulary(self):
        unique_chars = set()
        for text in self.corpus:
            for line in text.splitlines():
                for char in line:
                    unique_chars.add(char)
        self.vocab = sorted(list(unique_chars))
        self.vocab.append(self.end_of_word)

    def _pre_tokenize_corpus(self):
        word_splits = {}
        for text in self.corpus:
            for lines in text.splitlines():
                for word in lines.split():
                    if word:
                        char_list = list(word) + [self.end_of_word]
                        word_tuple = tuple(char_list)
                        word_splits[word_tuple] = word_splits.get(word_tuple, 0) + 1
        return word_splits

    def _get_pair_stats(self, splits):
        pair_counts = collections.defaultdict(int)
        for word_tuple, freq in splits.items():
            symbols = list(word_tuple)
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                pair_counts[pair] += freq
        return pair_counts

    def _merge_pair(self, pair_to_merge, splits):
        new_splits = {}
        (first, second) = pair_to_merge
        merged_token = first + second
        for word_tuple, freq in splits.items():
            symbols = list(word_tuple)
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(merged_token)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            new_splits[tuple(new_symbols)] = freq
        return new_splits

    def train(self, corpus):
        self.corpus = corpus
        self._initialize_vocabulary()
        current_splits = self._pre_tokenize_corpus()

        for i in range(self.num_merges):
            pair_stats = self._get_pair_stats(current_splits)
            if not pair_stats:
                break

            best_pair = max(pair_stats, key=pair_stats.get)
            current_splits = self._merge_pair(best_pair, current_splits)
            new_token = best_pair[0] + best_pair[1]
            self.vocab.append(new_token)
            self.merges[best_pair] = new_token
        
        self.vocab = sorted(list(set(self.vocab))) # Ensure unique and sorted
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def tokenize_word(self, word):
        tokens = list(word) + [self.end_of_word]
        
        while True:
            min_merge_pos = -1
            best_pair_to_merge = None
            
            for pair, new_token in self.merges.items():
                pos = -1
                for i in range(len(tokens) - 1):
                    if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                        pos = i
                        break
                
                if pos != -1 and (min_merge_pos == -1 or pos < min_merge_pos):
                    min_merge_pos = pos
                    best_pair_to_merge = pair
            
            if best_pair_to_merge is None:
                break
            
            first, second = best_pair_to_merge
            new_token = first + second
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == first and tokens[i+1] == second:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            
        return tokens

    def encode(self, text):
        encoded_ids = []
        for word in text.split():
            if word:
                word_tokens = self.tokenize_word(word)
                for token in word_tokens:
                    if token in self.token_to_id:
                        encoded_ids.append(self.token_to_id[token])
                    else:
                        # Handle unknown tokens, e.g., map to an <UNK> token or skip
                        # For simplicity, we'll skip for now or add a warning
                        print(f"Warning: Token '{token}' not in vocabulary.")
        return encoded_ids

    def decode(self, ids):
        tokens = [self.id_to_token.get(i, "<UNK>") for i in ids]
        text = "".join(tokens).replace(self.end_of_word, " ")
        return text.strip()

    def save_tokenizer(self, prefix):
        vocab_file = f"{prefix}_vocab.json"
        merges_file = f"{prefix}_merges.txt"

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.token_to_id, f, ensure_ascii=False, indent=2)

        with open(merges_file, 'w', encoding='utf-8') as f:
            for pair, new_token in self.merges.items():
                f.write(f"{pair[0]} {pair[1]}\n")

    def load_tokenizer(self, prefix):
        vocab_file = f"{prefix}_vocab.json"
        merges_file = f"{prefix}_merges.txt"

        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.token_to_id = json.load(f)
        
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.vocab = list(self.token_to_id.keys())

        with open(merges_file, 'r', encoding='utf-8') as f:
            self.merges = {}
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) == 2:
                    pair = (parts[0], parts[1])
                    self.merges[pair] = "".join(pair)

if __name__ == "__main__":
    corpus = [
        """I found me in a gloomy wood, astray
        Gone from the path direct: and e’en to tell
        It were no easy task, how savage wild
        That forest, how robust and rough its growth,
        Which to remember only, my dismay
        Renews, in bitterness not far from death.
        Yet to discourse of what there good befell,
        All else will I relate discover’d there.
        How first I enter’d it I scarce can say,
        Such sleepy dullness in that instant weigh’d
        My senses down, when the true path I left,
        But when a mountain’s foot I reach’d, where clos’d
        The valley, that had pierc’d my heart with dread,
        I look’d aloft, and saw his shoulders broad
        Already vested with that planet’s beam,
        Who leads all wanderers safe through every way."""
    ]

    print("Training corpus (line by line):")
    for text in corpus:
        for line in text.splitlines():
            print(line)

    tokenizer = BpeTokenizer(corpus=corpus, num_merges=15)

    print("\nInitial Vocabulary:")
    print(tokenizer.vocab)
    print(f"Vocabulary Size: {len(tokenizer.vocab)}")

    print("\nLearned Merges (Pair -> New Token):")
    for pair, token in tokenizer.merges.items():
        print(f"{pair} -> '{token}'")

    print("\nFinal Vocabulary (sorted):")
    print(tokenizer.vocab)

    print("\nToken to ID Mapping:")
    print(tokenizer.token_to_id)

    # --- Test the tokenizer ---
    test_sentence = "I found my path"
    print(f"\n--- Testing Tokenizer ---")
    print(f"Original Sentence: '{test_sentence}'")

    # Encode the sentence
    encoded_ids = tokenizer.encode(test_sentence)
    print(f"Encoded IDs: {encoded_ids}")

    # Decode the IDs
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Decoded Text: '{decoded_text}'")

    # Verify correctness
    print(f"Is decoding correct? {decoded_text == test_sentence}")

    # --- Test saving and loading ---
    print("\n--- Testing Save/Load --- ")
    tokenizer.save_tokenizer("test_tokenizer")
    
    new_tokenizer = BpeTokenizer()
    new_tokenizer.load_tokenizer("test_tokenizer")
    
    print("\nLoaded Tokenizer Vocab Size:", len(new_tokenizer.vocab))
    print("Loaded Tokenizer Merges:", len(new_tokenizer.merges))

    # Verify correctness of loaded tokenizer
    encoded_ids_new = new_tokenizer.encode(test_sentence)
    decoded_text_new = new_tokenizer.decode(encoded_ids_new)
    
    print(f"\nOriginal Sentence: '{test_sentence}'")
    print(f"Encoded IDs (from loaded tokenizer): {encoded_ids_new}")
    print(f"Decoded Text (from loaded tokenizer): '{decoded_text_new}'")
    print(f"Is decoding correct? {decoded_text_new == test_sentence}")
