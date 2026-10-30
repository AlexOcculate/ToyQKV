# Import the standard JSON library to handle text-based file saving
import json

# output file path
ofp = "model/tokenizer_config.json"

# Define a tiny dataset of raw text to train our tokenizer mapping
training_corpus = ["the river bank was muddy", "the money is in the bank"]

# Initialize an empty set to gather every unique character from the text library
unique_chars = set()

# Loop through each sentence in the training corpus
for sentence in training_corpus:
    # Loop through each individual character in the current sentence
    for char in sentence:
        # Add the character to the set (automatically handles deduplication)
        unique_chars.add(char)

# Convert the unique characters into a sorted list to ensure deterministic order
sorted_chars = sorted(list(unique_chars))

# Create a dictionary using a comprehension to map each character to a unique integer ID
char_to_id = {char: idx for idx, char in enumerate(sorted_chars)}

# Package the vocabulary dictionary into a configuration asset structure
tokenizer_config = {"vocab": char_to_id}

# Open the target file path in write mode to store the asset
with open(ofp, "w") as file:
    # Serialize the dictionary object into a formatted JSON string on disk
    json.dump(tokenizer_config, file, indent=4)
