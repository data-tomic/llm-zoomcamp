from fastembed import TextEmbedding
import numpy as np

supported_models = TextEmbedding.list_supported_models()

min_dimensionality = float('inf')
model_with_min_dim = None

print("Available models and their dimensionalities:")
for model_info in supported_models:
    dim = model_info.get('dim')
    model_name = model_info.get('model')
    print(f"Model: {model_name}, Dimensionality: {dim}")
    if dim is not None and dim < min_dimensionality:
        min_dimensionality = dim
        model_with_min_dim = model_name

if model_with_min_dim:
    print(f"\nSmallest dimensionality found: {min_dimensionality} (Model: {model_with_min_dim})")
else:
    print("\nCould not determine the smallest dimensionality.")