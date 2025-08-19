import os
import torch
import numpy as np
from utils.io import save_bin

# Fix: Use project root relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Define test directory
test_dir = os.path.join(PROJECT_ROOT, "kernels", "eltwise", "tests")
os.makedirs(test_dir, exist_ok=True)

# Generate data
shape = (64, 64)
A = torch.randn(shape).half().cpu().numpy()  # Save as FP16 NumPy
B = torch.randn(shape).half().cpu().numpy()
Out = A + B

# Save files
save_bin(A, os.path.join(test_dir, "input1.bin"))
save_bin(B, os.path.join(test_dir, "input2.bin"))
save_bin(Out, os.path.join(test_dir, "expected_output.bin"))

print(f"✅ Generated test data for eltwise add at: {test_dir}")