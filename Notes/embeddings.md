# Embeddings
## Prerequisites
 - It is recommended that you have gone through the `tokenization.md` file

## Overview
- So far, we’ve covered the process of tokenization — from having raw text, to defining tokenization rules, and finally assigning each character, subword, or word in a sentence an integer ID to represent it.

## Review
- In the `tokenizer.md` file,we explained what tokens are, but let’s briefly review them for example purposes.
- Tokens are small units of information, such as words, subwords, characters, images, or even segments of audio.
- Each token is then given an integer IDs,which are later embedded into vectors(list of numbers).
- These vectors:
   - encodes the meaning of that token.
   - Can be thought of as coordinates in a high-dimensional space, where tokens with similar meaning are positioned close to one another.
![tokens example:](../assets/tokens.png)


