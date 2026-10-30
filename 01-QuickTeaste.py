import numpy as np

# ==========================================
# 1. SETUP: Mocking your saved assets
# ==========================================
# Let's pretend this is your vocabulary dictionary
vocab = {"the": 0, "river": 1, "bank": 2, "was": 3, "muddy": 4}
uniqTokens = list(vocab.keys())

#
# Let's pretend your embedding layer uses
# - dimension size (d_model) of 4
#   - Rows match the vocab size (5),
#   - Columns match the embedding size (4)
#
np.random.seed(42)  # For reproducible random numbers
saved_embedding_layer = np.random.randn(5, 4)

# ==========================================
# 2. INPUT PROCESSING
# ==========================================
phrase = "the river bank was muddy"
tokens = phrase.split()

# Step A: Convert text tokens to IDs using your dictionary
token_ids = [vocab[token] for token in tokens]
print(f"Token IDs: {token_ids}")  # Output: [0, 1, 2, 3, 4]

# Step B: Look up the static vectors from your embedding matrix (Matrix 'X')
# X will have a shape of (Sequence Length, Embedding Dimension) -> (5, 4)
X = saved_embedding_layer[token_ids]
print(f"\nStatic Embeddings (X) Shape: {X.shape}")

# ==========================================
# 3. CREATING THE W_Q, W_K, W_V MATRICES
# ==========================================
# In a full model, these parameters are trained. Right now, we initialize them randomly.
# Their shape must be: (Input Embedding Dim, Output Attention Dim). Let's use 4 -> 4.
d_k = X.shape[1]  # Dimensionality factor = 4

W_Q = np.random.randn(4, 4)
W_K = np.random.randn(4, 4)
W_V = np.random.randn(4, 4)

# ==========================================
# 4. THE QKV ATTENTION PROCESS
# ==========================================

# Step 1: Project X into Queries, Keys, and Values
Q = np.dot(X, W_Q)  # Shape: (5, 4)
K = np.dot(X, W_K)  # Shape: (5, 4)
V = np.dot(X, W_V)  # Shape: (5, 4)

# Step 2: Compute raw alignment scores (Dot Product Q and K transposed)
# Shape will be (5, 4) x (4, 5) -> (5, 5) matrix of token-to-token scores
raw_scores = np.dot(Q, K.T)

# Step 3: Scale the scores to stabilize gradients
scaled_scores = raw_scores / np.sqrt(d_k)


# Step 4: Apply Softmax row-wise to turn scores into probability weights
def softmax(matrix):
    exp_matrix = np.exp(
        matrix - np.max(matrix, axis=-1, keepdims=True)
    )  # Stable softmax
    return exp_matrix / np.sum(exp_matrix, axis=-1, keepdims=True)


attention_weights = softmax(scaled_scores)
print("\n--- Attention Weights Matrix (5x5) ---")
print("Shows how much focus each word puts on every other word:")
print(np.round(attention_weights, 2))

# Step 5: Multiply weights by V to get context-rich vectors
# Shape: (5, 5) x (5, 4) -> (5, 4)
output_context_vectors = np.dot(attention_weights, V)

print(f"\nFinal Contextual Vectors Shape: {output_context_vectors.shape}")
print("These vectors now contain the blended meaning of the whole sentence!")
