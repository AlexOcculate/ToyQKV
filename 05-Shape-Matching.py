import json
import torch

ifp_model_config = "model/model_config.json"
ifp_model_weights = "model/model_weights.pt"

# 1. Read the user-supplied blueprint file (File B)
with open(ifp_model_config, "r") as f:
    config = json.load(f)

# 2. Open the heavy weight file header without fully mapping to memory layers yet (File C)
weights = torch.load(ifp_model_weights, weights_only=True)

# 3. Assert structural dimensions match perfectly before moving tensors to RAM
expected_embedding_shape = (config["vocab_size"], config["d_model"])
actual_embedding_shape = weights["wte.weight"].shape

if expected_embedding_shape != actual_embedding_shape:
    raise ValueError(
        f"CRITICAL DIMENSION MISMATCH: Config dictates embedding shape {expected_embedding_shape}, "
        f"but weights file contains shape {actual_embedding_shape}!"
    )

print("Shape Validation Passed: Assets are safe to materialize.")
