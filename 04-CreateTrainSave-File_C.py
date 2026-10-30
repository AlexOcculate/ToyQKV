# Import the PyTorch library to manage neural network operations and weight tensors
import torch

# Import the core neural network module from PyTorch to build layers
import torch.nn as nn

# Import the functional module from PyTorch to access mathematical helper functions
import torch.nn.functional as F

# output file path
ofp = "model/model_weights.pt"

# Define a mock dataset of token IDs matching the structure of our model configurations
dummy_input_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 5, 6, 0, 2]], dtype=torch.long)
# Define a mock dataset of target token IDs that the model should try to predict
dummy_targets = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 0, 2, 0]], dtype=torch.long)


# Create a custom class inherited from nn.Module to hold our minimal model graph
class MinimalTransformer(nn.Module):
    # Initialize the class constructor to define structural layers
    def __init__(self):
        # Call the parent class initializer to set up PyTorch tracking state
        super().__init__()
        # Instantiate the token embedding matrix table mapping 7 unique IDs to 4-dim vectors
        self.wte = nn.Embedding(7, 4)
        # Instantiate the Query, Key, and Value linear projection layer mapping 4 dims to 4 dims
        self.qkv_projection = nn.Linear(4, 4)
        # Instantiate the final language model head layer to map 4-dim outputs back to 7 vocab classes
        self.lm_head = nn.Linear(4, 7)

    # Define the forward pass method detailing how data flows through layers
    def forward(self, input_ids):
        # Pull static embedding vectors from the table using input token lookups
        embeddings = self.wte(input_ids)
        # Project static embedding vectors through the weight matrix into attention spaces
        qkv_states = self.qkv_projection(embeddings)
        # Map finished hidden states back to vocabulary scores using the output layer weights
        logits = self.lm_head(qkv_states)
        # Return computed logit scores representing next-token item distributions
        return logits


# Initialize an instance of our minimal transformer neural network model
model = MinimalTransformer()
# Initialize an AdamW optimization algorithm tracking our model's weight parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

# Loop through a set training phase cycle to run gradient adjustment updates
for epoch in range(5):
    # Reset all stored historical weight gradients to zero before computing new ones
    optimizer.zero_grad()
    # Execute a forward pass to calculate prediction scores from our raw inputs
    predictions = model(dummy_input_ids)
    # Calculate error scores by comparing flattened predictions against target distributions
    loss = F.cross_entropy(predictions.view(-1, 7), dummy_targets.view(-1))
    # Propagate errors backward through the network matrix graph to find parameter gradients
    loss.backward()
    # Adjust all weight matrix parameters based on current optimization gradient directions
    optimizer.step()

# Extract the fully optimized parameter state dictionary from the trained layers
trained_state = model.state_dict()
# Save the model weights dictionary to local storage as a reusable checkpoint asset
torch.save(trained_state, ofp)
