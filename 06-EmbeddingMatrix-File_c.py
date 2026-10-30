# Import the PyTorch library to access binary tensor loading functions
import torch

# Define the file path pointing to your saved master weights checkpoint (File C)
weights_file_path = "model/model_weights.pt"

print(f"Opening binary weights payload: {weights_file_path}...")

# Load the saved state dictionary dictionary safely from disk into system memory
# weights_only=True ensures we only unpickle structural tensor arrays for security
state_dict = torch.load(weights_file_path, weights_only=True)

# Define the target layer key name PyTorch automatically assigned to your embedding matrix
embedding_key = "wte.weight"

# Check if the specific embedding tensor exists within the loaded state dictionary keys
if embedding_key in state_dict:
    # Extract the raw tensor object matching our embedding key reference
    embedding_matrix = state_dict[embedding_key]

    print("\n[SUCCESS] Embedding Layer Matrix found safely inside File C!")
    print(
        f"Matrix Dimension Form (Vocab Size x d_model): {list(embedding_matrix.shape)}"
    )
    print("------------------------------------------------------------------------")

    # Print the absolute numerical float matrix values directly onto the console screen
    print(embedding_matrix)

    print("------------------------------------------------------------------------")
    print(
        "These numbers are the exact static coordinates used to translate incoming Token IDs."
    )

else:
    # Print an error diagnostic message highlighting available structure slots if mismatched
    print(
        f"\n[ERROR] Key '{embedding_key}' could not be located inside this weights checkpoint."
    )
    print(
        f"Available structural layers found inside this file: {list(state_dict.keys())}"
    )
