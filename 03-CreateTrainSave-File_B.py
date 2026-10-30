# Import the standard JSON library to handle text-based file saving
import json

# output file path
ofp = "model/model_config.json"


# Define the architectural dimensions for our custom Transformer model skeleton
model_architecture_spec = {
    "vocab_size": 50257,  # Define total unique tokens supported by the vocabulary
    "d_model": 768,  # Set the hidden vector dimensionality for layers and embeddings
    "n_heads": 12,  # Specify the number of parallel attention heads inside layers
    "n_layers": 12,  # Set the total number of stacked Transformer blocks to build
    "context_length": 1024,  # Set the maximum sequence token window for active processing
}

# Open the target file path in write mode to store the structural config asset
with open(ofp, "w") as file:
    # Serialize the specification dictionary into a formatted JSON string on disk
    json.dump(model_architecture_spec, file, indent=4)
