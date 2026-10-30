# Import the "math" module from Python's standard library for square-root and exponential helpers
import math

# Import the "json" module from Python's standard library to serialize weights to disk
import json

# Import the "random" module from Python's standard library for reproducible weight seeding
import random

# Import the "os" module so the checkpoint directory can be created before writing
import os

# Define the output file path where trained weights will be stored as plain text JSON
ofp = "model/model_weights.json"

# Set the vocabulary size: 7 unique token IDs exist in our mock dataset
VOCAB_SIZE = 7

# Set the embedding/hidden dimensionality used by every layer of the network
D_MODEL = 4

# Define a mock dataset of token IDs matching the structure of our model configurations
dummy_input_ids = [[0, 1, 2, 3, 4], [0, 5, 6, 0, 2]]

# Define a mock dataset of target token IDs that the model should try to predict
dummy_targets = [[1, 2, 3, 4, 0], [5, 6, 0, 2, 0]]

# Seed the random generator so every run produces identical initial weights and results
random.seed(42)


def make_matrix(rows, cols):
    # Allocate a nested list of shape (rows, cols) filled with zeros as placeholder storage
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def init_matrix(rows, cols):
    # Create a weight matrix whose entries are drawn from a small uniform random range
    return [[random.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]


def linear_forward(x, weight, bias):
    # Compute y = x @ W^T + b using the PyTorch convention: weight is [out_features][in_features]
    out_features = len(weight)
    # Read the input width from the matrix itself rather than trusting the caller's vector
    in_features = len(weight[0])
    # Fail loudly with a readable message when the incoming vector has the wrong width
    if len(x) != in_features:
        # Report which dimensions collided so the geometry bug is named, not guessed at
        raise ValueError(
            f"linear_forward: input vector has {len(x)} values but weight expects "
            f"{in_features} (weight shape {out_features}x{in_features})"
        )
    # Start each output accumulator at its corresponding bias value
    out = [bias[j] for j in range(out_features)]
    # Loop over every input feature and add its weighted contribution into each output
    for i, xi in enumerate(x):
        # Skip zero inputs early since they contribute nothing to the dot product sum
        if xi == 0.0:
            continue
        # Add weight[j][i] * x[i] into accumulator j for every output feature j
        for j in range(out_features):
            out[j] += xi * weight[j][i]
    # Return the finished output vector after all contributions have been summed
    return out


def embedding_lookup(table, token_id):
    # Return a fresh copy of the embedding row so callers cannot mutate the table directly
    return list(table[token_id])


def softmax(logits):
    # Find the maximum logit value to keep exponentials numerically stable
    max_logit = max(logits)
    # Exponentiate every centered logit into an unnormalized probability list
    exps = [math.exp(z - max_logit) for z in logits]
    # Sum all exponentials to obtain the partition (normalizing) constant
    total = sum(exps)
    # Divide each exponential by the total to produce a valid probability distribution
    return [e / total for e in exps]


def cross_entropy_loss_and_grad(logits_batch, targets_batch):
    # Count every token position across all rows so the mean matches torch's reduction='mean'
    n_positions = sum(len(row) for row in logits_batch)
    # Initialize the running loss accumulator at zero before any positions are scored
    total_loss = 0.0
    # Prepare a list of per-row gradient lists matching the nested shape of the logits
    grad_batch = []
    # Walk the logit rows and target rows together, one sequence at a time
    for row_logits, row_targets in zip(logits_batch, targets_batch):
        # Collect the per-position logit gradients belonging to this sequence row
        row_grads = []
        # Score each position against its own target token ID
        for logits, target in zip(row_logits, row_targets):
            # Convert raw scores into a probability distribution using the softmax function
            probs = softmax(logits)
            # Accumulate the negative log of the true-class probability into the total loss
            total_loss += -math.log(probs[target])
            # Build this position's gradient: (softmax - one_hot) divided by position count
            row_grads.append(
                [
                    (probs[j] - (1.0 if j == target else 0.0)) / n_positions
                    for j in range(len(probs))
                ]
            )
        # Append the finished gradient rows for this sequence into the batch collection
        grad_batch.append(row_grads)
    # Return the averaged scalar loss plus the nested per-position logit gradients
    return total_loss / n_positions, grad_batch


class MinimalTransformer:
    # Initialize the constructor to build every parameter tensor as plain nested lists
    def __init__(self):
        # Instantiate the token embedding table mapping 7 unique IDs to 4-dim vectors
        self.wte = init_matrix(VOCAB_SIZE, D_MODEL)
        # Instantiate the projection weight as [out_features=4][in_features=4] per nn.Linear
        self.qkv_weight = init_matrix(D_MODEL, D_MODEL)
        # Instantiate the projection bias vector with one small random value per output
        self.qkv_bias = [random.uniform(-0.5, 0.5) for _ in range(D_MODEL)]
        # FIX: store the head as nn.Linear(4, 7) does -> [out_features=7][in_features=4]
        self.lm_weight = init_matrix(VOCAB_SIZE, D_MODEL)
        # Instantiate the LM-head bias vector with one small random value per class
        self.lm_bias = [random.uniform(-0.5, 0.5) for _ in range(VOCAB_SIZE)]

    # Define the forward pass that turns integer token IDs into vocabulary logit scores
    def forward(self, input_ids):
        # Prepare a list of per-sequence logit rows collected during this forward traversal
        logits_batch = []
        # Loop over every token sequence (row) supplied in the batched input IDs
        for row in input_ids:
            # Prepare a list to collect the logit vector produced for each token position
            row_logits = []
            # Look up the embedding vector and run both linear layers for every token ID
            for token_id in row:
                # Pull the 4-dimensional embedding row for this token ID out of the table
                embedding = embedding_lookup(self.wte, token_id)
                # Project the embedding into the hidden "qkv" space via the linear layer
                qkv_state = linear_forward(embedding, self.qkv_weight, self.qkv_bias)
                # Map the hidden state back to vocabulary scores via the LM head layer
                logits = linear_forward(qkv_state, self.lm_weight, self.lm_bias)
                # Store the logit vector produced for this token position
                row_logits.append(logits)
            # Append the finished list of per-position logits for this sequence row
            logits_batch.append(row_logits)
        # Return computed logit scores representing next-token item distributions
        return logits_batch

    # Define a helper returning every trainable parameter in one fixed ordering
    def parameters(self):
        # Collect all weight matrices and bias vectors so gradients align by index
        return [self.wte, self.qkv_weight, self.qkv_bias, self.lm_weight, self.lm_bias]

    # Define a helper that writes an ordered parameter list back onto the model layers
    def set_parameters(self, values):
        # Unpack the ordered parameter list straight onto the five layer attributes
        self.wte, self.qkv_weight, self.qkv_bias, self.lm_weight, self.lm_bias = values


def assert_shapes(model):
    # Verify each tensor matches the architecture before any multiply touches it
    for name, tensor, rows, cols in [
        ("wte", model.wte, VOCAB_SIZE, D_MODEL),  # embedding table is vocab x hidden
        ("qkv_weight", model.qkv_weight, D_MODEL, D_MODEL),  # projection out x in
        ("lm_weight", model.lm_weight, VOCAB_SIZE, D_MODEL),  # head out(7) x in(4)
    ]:
        # Reject any tensor whose row count or row width drifted from the spec
        if len(tensor) != rows or len(tensor[0]) != cols:
            # Print the observed versus expected geometry for a one-glance diagnosis
            raise ValueError(
                f"{name}: got {len(tensor)}x{len(tensor[0])}, "
                f"expected {rows}x{cols}"
            )
    # Verify every bias vector matches its layer's number of output features
    for name, bias, size in [
        ("qkv_bias", model.qkv_bias, D_MODEL),  # one bias per projection output
        ("lm_bias", model.lm_bias, VOCAB_SIZE),  # one bias per vocabulary class
    ]:
        # Reject a bias whose length disagrees with its layer's row count
        if len(bias) != size:
            # Name the offending bias and both shapes involved
            raise ValueError(f"{name}: got {len(bias)} values, expected {size}")


def backward_step(model, input_ids, grad_batch):
    # Create zero-filled gradient containers matching each trainable parameter's shape
    grad_wte = make_matrix(VOCAB_SIZE, D_MODEL)
    # Projection gradients follow the same [out_features][in_features] layout as the weight
    grad_qkv_weight = make_matrix(D_MODEL, D_MODEL)
    # Projection bias gradients need one accumulator per projection output feature
    grad_qkv_bias = [0.0] * D_MODEL
    # FIX: head gradients use [out_features=7][in_features=4] to match lm_weight itself
    grad_lm_weight = make_matrix(VOCAB_SIZE, D_MODEL)
    # Head bias gradients need one accumulator per vocabulary class
    grad_lm_bias = [0.0] * VOCAB_SIZE
    # Loop over every sequence row in the batch to accumulate parameter gradients
    for row_ids, row_grad in zip(input_ids, grad_batch):
        # Recompute the forward activations for each position and fold them into gradients
        for token_id, logit_grad in zip(row_ids, row_grad):
            # Recompute the embedding vector for this token to reuse in backward math
            embedding = embedding_lookup(model.wte, token_id)
            # Recompute the qkv hidden state so head weight gradients can reference it
            qkv_state = linear_forward(embedding, model.qkv_weight, model.qkv_bias)
            # Accumulate head bias gradients and weight gradients in one pass per output j
            for j in range(VOCAB_SIZE):
                # The head bias gradient is simply the upstream logit gradient
                grad_lm_bias[j] += logit_grad[j]
                # The head weight gradient is the outer product of state and grad
                for i in range(D_MODEL):
                    grad_lm_weight[j][i] += logit_grad[j] * qkv_state[i]
            # Backpropagate through the head: dqkv[i] = sum_j dlogits[j] * W_head[j][i]
            dqkv = [0.0] * D_MODEL
            for i in range(D_MODEL):
                acc = 0.0
                for j in range(VOCAB_SIZE):
                    acc += logit_grad[j] * model.lm_weight[j][i]
                dqkv[i] = acc
            # Accumulate the projection bias gradient straight from the dqkv vector
            for i in range(D_MODEL):
                grad_qkv_bias[i] += dqkv[i]
            # Accumulate projection weight gradients as outer product of embedding x dqkv
            for j in range(D_MODEL):
                for i in range(D_MODEL):
                    grad_qkv_weight[j][i] += dqkv[j] * embedding[i]
            # Backpropagate through the projection: demb[i] = sum_j dqkv[j] * W_proj[j][i]
            dembed = [0.0] * D_MODEL
            for i in range(D_MODEL):
                acc = 0.0
                for j in range(D_MODEL):
                    acc += dqkv[j] * model.qkv_weight[j][i]
                dembed[i] = acc
            # Scatter the embedding gradient into this token's row of the table
            for k in range(D_MODEL):
                grad_wte[token_id][k] += dembed[k]
    # Return the finished gradient tensors in the same order as parameters()
    return [grad_wte, grad_qkv_weight, grad_qkv_bias, grad_lm_weight, grad_lm_bias]


def zeros_like(param):
    # Build a zero-filled structure matching either a matrix or a vector parameter
    if isinstance(param[0], list):
        return make_matrix(len(param), len(param[0]))
    return [0.0] * len(param)


def adamw_apply_vector(
    p_vec, g_vec, m_vec, v_vec, step, lr, beta1, beta2, eps, weight_decay
):
    # Allocate an updated scalar vector for the parameter values
    new_param = [0.0] * len(p_vec)
    # Allocate an updated scalar vector for the biased first moment estimate
    new_moment = [0.0] * len(p_vec)
    # Allocate an updated scalar vector for the biased second raw moment estimate
    new_var = [0.0] * len(p_vec)
    # Loop over every scalar coordinate of this vector parameter
    for i in range(len(p_vec)):
        # Compute the exponential moving average of the gradient with decay beta1
        m = beta1 * m_vec[i] + (1.0 - beta1) * g_vec[i]
        # Compute the exponential moving average of squared gradients with decay beta2
        v = beta2 * v_vec[i] + (1.0 - beta2) * g_vec[i] * g_vec[i]
        # Correct the first moment's initialization bias toward zero
        m_hat = m / (1.0 - beta1**step)
        # Correct the second moment's initialization bias toward zero
        v_hat = v / (1.0 - beta2**step)
        # Apply decoupled weight decay plus adaptive gradient scaling to update value
        new_param[i] = p_vec[i] - lr * (
            m_hat / (math.sqrt(v_hat) + eps) + weight_decay * p_vec[i]
        )
        # Store the freshly computed first moment for the next training step
        new_moment[i] = m
        # Store the freshly computed second moment for the next training step
        new_var[i] = v
    # Return updated parameter vector and both updated accumulator vectors
    return new_param, new_moment, new_var


def adamw_update(param, grad, moment, var, step, lr, beta1, beta2, eps, weight_decay):
    # FIX: dispatch matrices to a row loop calling the vector helper instead of recursing
    if isinstance(param[0], list):
        # Prepare lists that will hold updated rows for parameter and both accumulators
        new_param_rows, new_mom_rows, new_var_rows = [], [], []
        # Iterate over each corresponding row of param, grad, moment and var
        for p_row, g_row, m_row, v_row in zip(param, grad, moment, var):
            # Update this single row through the vector helper (no self-recursion)
            sub_p, sub_m, sub_v = adamw_apply_vector(
                p_row, g_row, m_row, v_row, step, lr, beta1, beta2, eps, weight_decay
            )
            # Collect the updated parameter row into the output matrix
            new_param_rows.append(sub_p)
            # Collect the updated moment row into the moment matrix
            new_mom_rows.append(sub_m)
            # Collect the updated variance row into the variance matrix
            new_var_rows.append(sub_v)
        # Return the rebuilt matrix parameter plus its two rebuilt accumulators
        return new_param_rows, new_mom_rows, new_var_rows
    # Delegate plain vector parameters directly to the scalar update helper
    return adamw_apply_vector(
        param, grad, moment, var, step, lr, beta1, beta2, eps, weight_decay
    )


# Initialize an instance of our minimal transformer neural network model
model = MinimalTransformer()

# Validate all tensor geometry once before training touches any of them
assert_shapes(model)

# Create zero-filled first-moment (mean) accumulators, one per parameter tensor
momentums = [zeros_like(p) for p in model.parameters()]

# Create zero-filled second-moment (uncentered variance) accumulators per parameter
variances = [zeros_like(p) for p in model.parameters()]

# Configure the AdamW learning rate hyper-parameter to match the original script
lr = 0.01

# Configure the beta1 decay factor controlling the first moment's momentum speed
beta1 = 0.9

# Configure the beta2 decay factor controlling the second moment's variance speed
beta2 = 0.999

# Configure epsilon to prevent division by zero inside the adaptive gradient term
eps = 1e-8

# Configure decoupled weight decay strength applied multiplicatively each step
weight_decay = 0.01

# Initialize a step counter used for bias correction of the moment estimates
step = 0

# Loop through a set training phase cycle to run gradient adjustment updates
for epoch in range(5):
    # Execute a forward pass to calculate prediction scores from our raw inputs
    logits_batch = model.forward(dummy_input_ids)
    # Compute mean cross-entropy loss and per-position gradients in nested form
    loss, grad_batch = cross_entropy_loss_and_grad(logits_batch, dummy_targets)
    # Print a progress line showing the epoch number and current average loss
    print(f"epoch {epoch}, loss {loss:.6f}")
    # Accumulate parameter gradients from every token position in the batch
    grads = backward_step(model, dummy_input_ids, grad_batch)
    # Increment the optimizer step counter used to bias-correct moment estimates
    step += 1
    # Prepare containers for updated parameters and their two moment accumulators
    new_params, new_momentums, new_variances = [], [], []
    # Loop over every parameter with its gradient and running moment accumulators
    for p, g, m, v in zip(model.parameters(), grads, momentums, variances):
        # Apply one AdamW update to this parameter tensor and get fresh values back
        upd_p, upd_m, upd_v = adamw_update(
            p, g, m, v, step, lr, beta1, beta2, eps, weight_decay
        )
        # Collect the updated parameter tensor for reassignment after the loop
        new_params.append(upd_p)
        # Collect the updated first-moment accumulator tensor
        new_momentums.append(upd_m)
        # Collect the updated second-moment accumulator tensor
        new_variances.append(upd_v)
    # Write the freshly updated parameter values back into the model's attributes
    model.set_parameters(new_params)
    # Replace momentum accumulators with their updated versions for the next step
    momentums = new_momentums
    # Replace variance accumulators with their updated versions for the next step
    variances = new_variances

# Extract the fully optimized parameter state dictionary from the trained layers
trained_state = {
    "wte": model.wte,  # embedding table with shape vocab_size x d_model
    "qkv_weight": model.qkv_weight,  # projection weight with shape out x in
    "qkv_bias": model.qkv_bias,  # projection bias with length d_model
    "lm_weight": model.lm_weight,  # head weight with shape vocab_size x d_model
    "lm_bias": model.lm_bias,  # head bias with length vocab_size
}

# Build the parent directory path for the checkpoint if it does not exist yet
os.makedirs(os.path.dirname(ofp) or ".", exist_ok=True)

# Save the model weights dictionary to local storage as a reusable JSON asset
with open(ofp, "w") as f:
    json.dump(trained_state, f, indent=2)
