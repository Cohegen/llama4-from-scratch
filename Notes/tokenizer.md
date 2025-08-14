# Tokenization

## Overview
- Tokenization is the process of converting raw text into a sequence of tokens(numbers) that a Large Language Model can process.
- These tokens can represent characters,subwords, or words depending on the tokenization strategy.
 

### Input
A raw sentence, for example: 
"This is an input text."

## Output:
A sequence of integers that correspond to tokens in the tokenizer's vocabulary.
The process of tokenization is shown in the diagram below:
![Output Example](../assets/tokenization.png)

## Tokenizer 
Here we' ll use byte pair encoding.
The code implemetation of Byte Pair Encoding is in `src/tokenizer.py`.
### Byte Pair Encoding
1. Training Corpus
   - The algorithm starts with a training corpus, in case a passes from `The Divine Comedy` by Dante Aliegheri
   - ![Dante](../assets/dante.jpg).
   - This text is the source from which the vocabulary and merge rule will be runned.
     




