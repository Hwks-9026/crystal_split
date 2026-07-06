import torch
import numpy as np
import diffraction_sim

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Test the Rust generator (using a small 256x256 image size)
# Note: Ensure you removed the hardcoded `let img_size = 1500;` from your Rust code!
composite_bytes, mask_bytes_list = diffraction_sim.generate_sample(256)

print(f"Composite image raw bytes length: {len(composite_bytes)}")
print(f"Number of masks generated: {len(mask_bytes_list)}")
print("Environment setup is successful!")
