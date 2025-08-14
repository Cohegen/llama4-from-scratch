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

# Initialize vocabulary with unique characters
unique_chars = set()
for text in corpus:
    for line in text.splitlines():
        for char in line:
            unique_chars.add(char)

vocab = list(unique_chars)
vocab.sort()  # consistent order

# Adding a special end-of-word token
end_of_word = "</w>"
vocab.append(end_of_word)

print("\nInitial Vocabulary:")
print(vocab)
print(f"Vocabulary Size: {len(vocab)}")

# Pre-tokenizing the corpus: splitting into words, then into characters
word_splits = {}
for text in corpus:
    for lines in text.splitlines():
        for word in lines.split():  #split into words
            if word:
                char_list = list(word) + [end_of_word]
                word_tuple = tuple(char_list)  # immutable for dictionary keys
                word_splits[word_tuple] = word_splits.get(word_tuple, 0) + 1

print("\nWord splits with frequencies:")
for k, v in word_splits.items():
    print(k, ":", v)

import collections

def get_pair_stats(splits):
    """A function that counts the frequency of adjacent paits in the word splits"""
    pair_counts = collections.defaultdict(int)
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        for i in range(len(symbols)-1):
            pair = (symbols[i],symbols[i+1])
            pair_counts[pair] += freq ##adding the frequency of the word to the pair count
    return pair_counts

def merge_pair(pair_to_merge,splits):
    """Meges the specified pair in the word splits."""
    new_splits = {}
    (first,second) = pair_to_merge
    merged_token = first + second
    for word_tuple, freq in splits.items():
        symbols = list(word_tuple)
        new_symbols= []
        i = 0
        while i < len(symbols):
            #if current and next symbol match the pair to merge
            if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                new_symbols.append(merged_token)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_splits[tuple(new_symbols)] = freq # use the updated symbol list as the key
    return new_splits


##BPE training intialization
num_merges = 15
merges = {}

current_splits = word_splits.copy() #start with initial word splits

print("\n Starting BPE merges:")
print(f"Initial Splits: {current_splits}")
print("-"*30)

for i in range(num_merges):
    print(f"\nMerge Iteration {i+1}/{num_merges}")

    #calculating pair frequencies
    pair_stats = get_pair_stats(current_splits)
    if not pair_stats:
        print("No more pairs to merge.")
        break

    #finding best pair
    best_pair = max(pair_stats,key=pair_stats.get)
    best_freq = pair_stats[best_pair]
    print(f"Found Best Pair: {best_pair} with Frequency : {best_freq}")

    #merge the best pair
    current_splits = merge_pair(best_pair,current_splits)
    new_token = best_pair[0] + best_pair[1]
    print(f"Merging {best_pair} into '{new_token}'")
    print(f"Splits after merge: {current_splits}")

    #update vocabulary
    vocab.append(new_token)
    print(f"Updated Vocabluary: {vocab}")

    # store merge rule
    merges[best_pair] = new_token
    print(f"Updated Merges: {merges}")

    print("-"*30)


# BPE Merges Complete 
print("\n--- BPE Merges Complete ---")
print(f"Final Vocabulary Size: {len(vocab)}")
print("\nLearned Merges (Pair -> New Token):")
# Pretty print merges
for pair, token in merges.items():
    print(f"{pair} -> '{token}'")

print("\nFinal Word Splits after all merges:")
print(current_splits)

print("\nFinal Vocabulary (sorted):")
# Sort for consistent viewing
final_vocab_sorted = sorted(list(set(vocab))) # Use set to remove potential duplicates if any step introduced them
print(final_vocab_sorted)
