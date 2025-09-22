# python/run.py
import os
import sys
import importlib.util
import torch
import time
import numpy as np
import struct
import plotly.graph_objects as go
import plotly.io as pio
import webbrowser
from tabulate import tabulate
from python.utils.io import load_fp16, load_bf16
import argparse
import torch.nn.functional as F

# --- Project Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
KERNELS_DIR = os.path.join(PROJECT_ROOT, "kernels")

from python.reference import ref

def visualize_enhanced_errors(actual, expected, kernel_name, test_name, data_handler, actual_u16_raw=None):
    """
    Enhanced visualization showing hex values and detailed diffs for mismatches
    """
    diff = np.abs(actual - expected)
    error_mask = diff > (data_handler["atol"] + data_handler["rtol"] * np.abs(expected))
    
    # Convert to uint16 representations
    expected_u16 = np.frombuffer(expected.tobytes(), dtype=np.uint16).reshape(expected.shape)
    if actual_u16_raw is not None:
        actual_u16 = actual_u16_raw.astype(np.uint16).reshape(expected.shape)
    else:
        # Fallback for BF16
        actual_u16 = (actual.view(np.uint32) >> 16).astype(np.uint16)
    
    xor_vals = expected_u16 ^ actual_u16
    
    # Find mismatches
    mismatch_indices = np.where(error_mask)
    mismatch_count = len(mismatch_indices[0])
    
    if mismatch_count == 0:
        return
    
    print(f"\n=== ENHANCED ERROR ANALYSIS FOR {kernel_name}/{test_name} ===")
    print(f"Total mismatches: {mismatch_count}")
    
    # Show top 10 worst mismatches with full details
    flat_diff = diff.flatten()
    mismatch_flat_indices = np.where(error_mask.flatten())[0]
    top_indices = np.argsort(flat_diff[mismatch_flat_indices])[-10:][::-1]
    top_original_indices = mismatch_flat_indices[top_indices]
    
    for i, flat_idx in enumerate(top_original_indices):
        multi_idx = np.unravel_index(flat_idx, diff.shape)
        exp_val = expected[multi_idx]
        act_val = actual[multi_idx]
        exp_u16 = expected_u16[multi_idx]
        act_u16 = actual_u16[multi_idx]
        xor_val = xor_vals[multi_idx] if xor_vals.size > 1 else xor_vals.item()
        abs_diff = abs(act_val - exp_val)
        
        print(f"\n--- Mismatch #{i+1} ---")
        print(f"Position: {multi_idx}")
        print(f"Expected: {exp_val:.8f} (0x{exp_u16:04X})")
        print(f"Actual:   {act_val:.8f} (0x{act_u16:04X})")
        print(f"Diff:     {abs_diff:.8f}")
        print(f"XOR:      0x{xor_val:04X}")
        
        # Show bit patterns
        exp_bits = f"{exp_u16:016b}"
        act_bits = f"{act_u16:016b}"
        print(f"Exp bits: {exp_bits}")
        print(f"Act bits: {act_bits}")
        
        # Highlight differing bits
        xor_bits = f"{xor_val:016b}"
        highlighted = ""
        for j, (e_bit, a_bit, x_bit) in enumerate(zip(exp_bits, act_bits, xor_bits)):
            if x_bit == '1':
                highlighted += f"[{e_bit}{a_bit}]"
            else:
                highlighted += f" {e_bit} "
            if (j + 1) % 4 == 0 and j < 15:
                highlighted += "|"
        print(f"Diff vis: {highlighted}")

class ErrorAnalyzer:
    """
    Analyzes and reports mismatches between expected and actual outputs.
    Generates plots, detailed text reports, and detects patterns.
    """

    def __init__(self, actual, expected, test_name, kernel_name, test_path, data_handler, actual_u16_raw=None):
        self.actual = actual
        self.expected = expected
        self.test_name = test_name
        self.kernel_name = kernel_name
        self.test_path = test_path
        self.data_handler = data_handler
        self.atol = data_handler["atol"]
        self.rtol = data_handler["rtol"]

        self.diff = np.abs(actual - expected)
        self.mismatch_mask = self.diff > (self.atol + self.rtol * np.abs(expected))
        self.mismatch_indices = np.where(self.mismatch_mask)
        self.mismatch_count = len(self.mismatch_indices[0])
        self.total_count = actual.size

        # Flatten for easier analysis
        self.flat_actual = actual.flatten()
        self.flat_expected = expected.flatten()
        self.flat_diff = self.diff.flatten()
        self.flat_indices = list(zip(*[idx.flatten() for idx in self.mismatch_indices]))

        # For bit-level analysis
        self.expected_u16 = np.frombuffer(expected.tobytes(), dtype=np.uint16).reshape(expected.shape)
        if actual_u16_raw is not None:
            self.actual_u16 = actual_u16_raw.astype(np.uint16).reshape(expected.shape)
        else:
        # Fallback (may be incorrect for BF16!)
            self.actual_u16 = (actual.view(np.uint32) >> 16).astype(np.uint16)
        self.xor_vals = self.expected_u16 ^ self.actual_u16

    def generate_enhanced_report(self):
        """Generate an enhanced report with hex values and bit-level analysis"""
        if self.mismatch_count == 0:
            return
            
        print(f"\n=== DETAILED HEX ANALYSIS FOR {self.kernel_name}/{self.test_name} ===")
        
        # Get top 15 worst mismatches
        flat_diff_mismatch = self.flat_diff[np.where(self.mismatch_mask.flatten())[0]]
        top_indices_local = np.argsort(flat_diff_mismatch)[-15:][::-1]
        top_original_flat_indices = np.where(self.mismatch_mask.flatten())[0][top_indices_local]
        
        for i, flat_idx in enumerate(top_original_flat_indices):
            multi_idx = np.unravel_index(flat_idx, self.diff.shape)
            exp_val = self.expected[multi_idx]
            act_val = self.actual[multi_idx]
            exp_u16 = self.expected_u16[multi_idx]
            act_u16 = self.actual_u16[multi_idx]
            xor_val = self.xor_vals[multi_idx] if self.xor_vals.size > 1 else self.xor_vals.item()
            abs_diff = abs(act_val - exp_val)
            
            print(f"\n#{i+1} Position {multi_idx}:")
            print(f"  Float: {exp_val:12.6f} vs {act_val:12.6f} (diff: {abs_diff:.2e})")
            print(f"  Hex:   0x{exp_u16:04X} vs 0x{act_u16:04X} (XOR: 0x{xor_val:04X})")
            
            # Bit-level comparison
            exp_bits = f"{exp_u16:016b}"
            act_bits = f"{act_u16:016b}"
            print(f"  Exp:   {exp_bits}")
            print(f"  Act:   {act_bits}")
            
            # Visualize bit differences
            diff_vis = ""
            for j, (e_bit, a_bit) in enumerate(zip(exp_bits, act_bits)):
                if e_bit != a_bit:
                    diff_vis += "^"
                else:
                    diff_vis += " "
                if (j + 1) % 4 == 0 and j < 15:
                    diff_vis += "|"
            print(f"  Diff:  {diff_vis}")

    def generate_report(self, full_diff=False):
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f" ERROR ANALYSIS REPORT: {self.kernel_name} / {self.test_name}")
        report_lines.append("=" * 80)
        report_lines.append("")

        # --- Summary ---
        mismatch_percentage = (self.mismatch_count / self.total_count) * 100
        max_error = np.max(self.diff)
        mean_abs_error = np.mean(self.diff)
        avg_error_on_mismatch = np.mean(self.diff[self.mismatch_mask]) if self.mismatch_count > 0 else 0.0

        report_lines.append("[SUMMARY]")
        report_lines.append(f"  - Total Elements: {self.total_count}")
        report_lines.append(f"  - Mismatched Elements: {self.mismatch_count} ({mismatch_percentage:.4f}%)")
        report_lines.append(f"  - Max Absolute Error: {max_error:.8f}")
        report_lines.append(f"  - Mean Absolute Error: {mean_abs_error:.8f}")
        report_lines.append(f"  - Avg Error on Mismatches: {avg_error_on_mismatch:.8f}")
        report_lines.append("")

        # --- Error Distribution ---
        if self.mismatch_count > 0:
            report_lines.append("[ERROR DISTRIBUTION]")
            percentiles = [50, 90, 95, 99]
            for p in percentiles:
                val = np.percentile(self.diff[self.mismatch_mask], p)
                report_lines.append(f"  - {p}th Percentile Error: {val:.8f}")
            report_lines.append("")

        # --- Pattern Detection ---
        patterns = self.detect_patterns()
        if patterns:
            report_lines.append("[DETECTED PATTERNS]")
            for pattern in patterns:
                report_lines.append(f"  - {pattern}")
            report_lines.append("")

        # --- Top 5 Worst Mismatches ---
        report_lines.append("[TOP 5 WORST MISMATCHES]")
        if self.mismatch_count > 0:
            # Get indices of top 5 largest errors
            mismatch_flat_indices = np.where(self.mismatch_mask.flatten())[0]
            flat_diff_mismatch = self.flat_diff[mismatch_flat_indices]
            # 3. Get the sort order for this smaller list of 18 errors
            top5_local_indices = np.argsort(flat_diff_mismatch)[-5:][::-1]
            # 4. Use this sort order to select the correct original flat indices
            top5_original_flat_indices = mismatch_flat_indices[top5_local_indices]

            for i, flat_idx in enumerate(top5_original_flat_indices):
                # Now flat_idx is a correct index into the full tensor
                multi_idx = np.unravel_index(flat_idx, self.diff.shape)
                exp_val = self.expected[multi_idx]
                act_val = self.actual[multi_idx]
                exp_u16 = self.expected_u16[multi_idx]
                act_u16 = self.actual_u16[multi_idx]
                xor_val = exp_u16 ^ act_u16
                xor_str = "".join(['^' if (xor_val >> j) & 1 else '.' for j in range(15, -1, -1)])
                def get_bits(u16_val):
                    s = (u16_val >> 15) & 1
                    e = (u16_val >> 7) & 0xFF
                    m = u16_val & 0x7F
                    return f"S:{s} E:{e:08b} M:{m:07b}"
                exp_val_real = (np.array([exp_u16], dtype=np.uint32) << 16).view(np.float32)[0]
                act_val_real = (np.array([act_u16], dtype=np.uint32) << 16).view(np.float32)[0]
                diff = abs(act_val_real - exp_val_real)

                # --- Modify the print statements ---
                report_lines.append(f"\n  #{i+1} Index {str(multi_idx).replace('np.int64', ''):<20}")
                report_lines.append(f"    - Float: {exp_val_real:<15.8f} vs {act_val_real:<15.8f} (Diff: {diff:.8f})") # Changed
                report_lines.append(f"    - Hex:   0x{exp_u16:04X} vs 0x{act_u16:04X}")
                report_lines.append(f"    - Bits (Exp): {get_bits(exp_u16)}")
                report_lines.append(f"    - Bits (Act): {get_bits(act_u16)}")
                report_lines.append(f"    - XOR:   {xor_str}")
        else:
            report_lines.append("  - No mismatches to show.")
        report_lines.append("")

        # --- Full Mismatch Dump (if requested) ---
        if full_diff and self.mismatch_count > 0:
            report_lines.append(f"[FULL MISMATCH LIST: {self.mismatch_count} entries]")
            for i, flat_idx in enumerate(np.where(self.mismatch_mask.flatten())[0]):
                multi_idx = np.unravel_index(flat_idx, self.diff.shape)
                exp_val = self.expected[multi_idx]
                act_val = self.actual[multi_idx]
                exp_u16 = self.expected_u16[multi_idx]
                act_u16 = self.actual_u16[multi_idx]
                xor_val = exp_u16 ^ act_u16
                xor_str = "".join(['^' if (xor_val >> j) & 1 else '.' for j in range(15, -1, -1)])

                report_lines.append(f"\n  [{i+1:05d}] Index {str(multi_idx):<20}")
                report_lines.append(f"    - Float: {exp_val:<15.8f} vs {act_val:<15.8f}")
                report_lines.append(f"    - Hex:   0x{exp_u16:04X} vs 0x{act_u16:04X}")
                report_lines.append(f"    - XOR:   {xor_str}")

        # Write to file
        os.makedirs("reports", exist_ok=True)
        safe_name = f"{self.kernel_name}_{self.test_name}".replace("/", "_")
        report_path = os.path.join("reports", f"{safe_name}_error_report.txt")
        abs_report_path = os.path.abspath(report_path)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"  -> Detailed error report saved to {abs_report_path}")

        # Also print summary to console
        print("\n" + "\n".join(report_lines[:20]))  # Print first 20 lines to console
        if len(report_lines) > 20:
            print("  ... (full report in error_report.txt)")

    def detect_patterns(self):
        patterns = []

        if self.mismatch_count == 0:
            return patterns

        # --- Pattern 1: All mismatches in last row/column (edge effect) ---
        if self.actual.ndim >= 2:
            last_row_mismatch = np.any(self.mismatch_mask[-1, :])
            last_col_mismatch = np.any(self.mismatch_mask[:, -1])
            first_row_mismatch = np.any(self.mismatch_mask[0, :])
            first_col_mismatch = np.any(self.mismatch_mask[:, 0])

            if last_row_mismatch and not np.any(self.mismatch_mask[:-1, :]):
                patterns.append("All mismatches occur in the last row (possible boundary handling bug).")
            if last_col_mismatch and not np.any(self.mismatch_mask[:, :-1]):
                patterns.append("All mismatches occur in the last column (possible boundary handling bug).")
            if first_row_mismatch and not np.any(self.mismatch_mask[1:, :]):
                patterns.append("All mismatches occur in the first row (possible padding/offset bug).")
            if first_col_mismatch and not np.any(self.mismatch_mask[:, 1:]):
                patterns.append("All mismatches occur in the first column (possible padding/offset bug).")

        # --- Pattern 2: All mismatches have same bit flipped ---
        xor_flat = self.xor_vals[self.mismatch_mask]
        if len(xor_flat) > 0:
            unique_xors = np.unique(xor_flat)
            if len(unique_xors) == 1:
                bit_pos = int(np.log2(unique_xors[0])) if unique_xors[0] > 0 and (unique_xors[0] & (unique_xors[0]-1)) == 0 else None
                if bit_pos is not None:
                    patterns.append(f"All mismatches flip bit {bit_pos} (possible bit-shift or mask error).")
                else:
                    patterns.append(f"All mismatches have identical XOR pattern: 0x{unique_xors[0]:04X} (systematic corruption).")

        # --- Pattern 3: Errors are always positive or always negative ---
        signed_diff = self.actual - self.expected
        signed_diff_mismatch = signed_diff[self.mismatch_mask]
        if np.all(signed_diff_mismatch > 0):
            patterns.append("All errors are positive (systematic overestimation).")
        elif np.all(signed_diff_mismatch < 0):
            patterns.append("All errors are negative (systematic underestimation).")

        # --- Pattern 4: Mismatches clustered in blocks (e.g., 16x16) ---
        if self.actual.ndim >= 2:
            block_size = 16
            h, w = self.actual.shape[:2]
            block_errors = np.zeros((h//block_size + 1, w//block_size + 1), dtype=int)
            for r in range(h):
                for c in range(w):
                    if np.any(self.mismatch_mask[r, c]):
                        br, bc = r // block_size, c // block_size
                        block_errors[br, bc] += 1

            max_block_err = np.max(block_errors)
            total_blocks = block_errors.size
            blocks_with_errors = np.count_nonzero(block_errors)
            if max_block_err > 0 and blocks_with_errors < total_blocks // 2:
                patterns.append(f"Mismatches are clustered in {blocks_with_errors}/{total_blocks} blocks (possible tile/thread mapping issue).")

        return patterns

# --- Operator Abstraction ---
class Operator:
    """
    Base class for all operators.
    """
    def get_pytorch_baseline(self, a, b):
        raise NotImplementedError

    def get_kernel_call(self, kernel_lib, a, b, shape, expected_shape, kernel_name):
        raise NotImplementedError

class EltwiseAdd(Operator):
    """
    Element-wise addition operator.
    """
    def get_pytorch_baseline(self, a, b):
        return ref.eltwise_add(a, b)

    def get_kernel_call(self, kernel_lib, a, b, shape, expected_shape, kernel_name):
        if "fp16" in kernel_name:
            return kernel_lib.eltwise_add_fp16_cu(a, b)
        elif "bf16" in kernel_name:
            return kernel_lib.eltwise_add_bf16_cu(a, b)
        else:
            raise ValueError("Unsupported data type for EltwiseAdd")


class MatMul(Operator):
    """
    Matrix multiplication operator.
    """
    def get_pytorch_baseline(self, a, b):
        return ref.matmul_fp16(a, b) if a.dtype == torch.float16 else ref.matmul_bf16(a, b)

    def get_kernel_call(self, kernel_lib, a, b, shape, expected_shape, kernel_name):
        m, n, k = shape[0], expected_shape[1], shape[1]
        if "fp16" in kernel_name:
            return kernel_lib.matmul_fp16_cu(a, b, m, n, k)
        elif "bf16" in kernel_name:
            return kernel_lib.matmul_bf16_cu(a, b, m, n, k)
        else:
            raise ValueError("Unsupported data type for MatMul")

# --- Operator Registry ---

class Conv2D(Operator):
    def get_pytorch_baseline(self, inputs, params):
        return F.conv2d(inputs["X"], inputs["Weights"], stride=params["stride"], padding=params["padding"])

    def get_kernel_call(self, kernel_lib, inputs, params, kernel_name):
        p = params # shortcut
        h_out = (p["H"] - p["R"] + 2 * p["padding"]) // p["stride"] + 1
        w_out = (p["W"] - p["S"] + 2 * p["padding"]) // p["stride"] + 1
        output_shape = (p["N"], p["K"], h_out, w_out)
        out = np.zeros(output_shape, dtype=np.uint16)

        if "fp16" in kernel_name:
            kernel_lib.conv2d_fp16_cu(
                inputs["X"], inputs["Weights"], out,
                p["N"], p["C"], p["H"], p["W"],
                p["K"], p["R"], p["S"],
                p["stride"], p["padding"]
            )
        elif "bf16" in kernel_name:
            kernel_lib.conv2d_bf16_cu(
                inputs["X"], inputs["Weights"], out,
                p["N"], p["C"], p["H"], p["W"],
                p["K"], p["R"], p["S"],
                p["stride"], p["padding"]
            )
        else:
            raise ValueError("Unsupported data type for Conv2D")
        return out


OPERATOR_REGISTRY = {
    "eltw_add": EltwiseAdd(),
    "matmul": MatMul(),
    "conv2d": Conv2D(),
}

# --- Plotting Utility ---
def visualize_tensor_errors(actual, expected, kernel_name, test_name, args=None, data_handler=None, actual_u16_raw=None):
    """
    Enhanced visualization with hover information showing hex values
    """
    # Compute diff and error mask (using same tolerance as test)
    diff = np.abs(actual - expected)
    # Use a fixed tolerance for visualization — you can also pass atol/rtol if needed
    tolerance = data_handler["atol"] + data_handler["rtol"] * np.abs(expected)
    error_mask = diff > tolerance  # True = mismatch (red), False = match (green)
    error_mask_uint8 = error_mask.astype(np.uint8)

    # Save binary mask (backward compatible)
    if error_mask_uint8.ndim > 2:
        plot_shape = (-1, error_mask_uint8.shape[-2], error_mask_uint8.shape[-1])
        error_mask_flat = error_mask_uint8.reshape(plot_shape)
    else:
        error_mask_flat = error_mask_uint8

    mask_file_path = os.path.join("reports", f"{kernel_name}_{test_name}_error_mask.bin")
    os.makedirs("reports", exist_ok=True)
    with open(mask_file_path, "wb") as f:
        if error_mask_flat.ndim == 3:
            f.write(struct.pack("<III", error_mask_flat.shape[0], error_mask_flat.shape[1], error_mask_flat.shape[2]))
        else:
            f.write(struct.pack("<II", error_mask_flat.shape[0], error_mask_flat.shape[1]))
        f.write(error_mask_flat.tobytes())

    # Prepare hover text with detailed information
    expected_u16 = np.frombuffer(expected.tobytes(), dtype=np.uint16).reshape(expected.shape)
    if actual_u16_raw is not None:
        actual_u16 = actual_u16_raw.astype(np.uint16).reshape(expected.shape)
    else:
        actual_u16 = (actual.view(np.uint32) >> 16).astype(np.uint16)
    
    hover_text = np.empty(actual.shape, dtype=object)
    for idx in np.ndindex(actual.shape):
        if error_mask[idx]:
            exp_val = expected[idx]
            act_val = actual[idx]
            exp_hex = expected_u16[idx]
            act_hex = actual_u16[idx]
            xor_val = exp_hex ^ act_hex
            abs_diff = abs(act_val - exp_val)
            hover_text[idx] = (
                f"Position: {idx}<br>"
                f"Expected: {exp_val:.6f} (0x{exp_hex:04X})<br>"
                f"Actual:   {act_val:.6f} (0x{act_hex:04X})<br>"
                f"Diff:     {abs_diff:.2e}<br>"
                f"XOR:      0x{xor_val:04X}"
            )
        else:
            hover_text[idx] = f"Position: {idx}<br>Match: OK"

    # --- PLOTLY INTERACTIVE VISUALIZATION ---
    title = f"{kernel_name}/{test_name} Pass/Fail Map"

    fig = go.Figure()

    # Binary green/red colormap
    colorscale = [(0, "green"), (1, "red")]

    if actual.ndim <= 2:
        fig.add_trace(go.Heatmap(
            z=error_mask.astype(int),
            text=hover_text,
            hoverinfo='text',
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=False,
            name='Pass/Fail'
        ))
        fig.update_layout(title=f"{title}")

    else:
        # Assume last two dims are spatial (H, W)
        spatial_shape = actual.shape[-2:]
        outer_shape = actual.shape[:-2]
        total_slices = np.prod(outer_shape)

        error_slices = error_mask.reshape(-1, *spatial_shape).astype(int)
        hover_slices = hover_text.reshape(-1, *spatial_shape)

        # Add traces for first slice
        fig.add_trace(go.Heatmap(
            z=error_slices[0],
            text=hover_slices[0],
            hoverinfo='text',
            colorscale=colorscale,
            zmin=0,
            zmax=1,
            showscale=False,
            name='Pass/Fail'
        ))

        # Slider for slices
        steps = []
        for i in range(total_slices):
            step = dict(
                method="update",
                args=[{"z": [error_slices[i]], "text": [hover_slices[i]]}],  # Update z data for heatmap
                label=f"Slice {i}"
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Slice: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders,
            title=f"{title} - Slice 0",
            xaxis_title="Columns",
            yaxis_title="Rows",
            yaxis_autorange="reversed"
        )

    # Save HTML
    safe_name = f"{kernel_name}_{test_name}".replace("/", "_").replace(" ", "_")
    out_file = os.path.join("reports", f"{safe_name}_pass_fail_viz.html")
    abs_out_file = os.path.abspath(out_file)

    pio.write_html(fig, file=out_file, auto_open=False)
    print(f"  -> Pass/Fail visualization saved to {abs_out_file}")
# --- Data Handling ---

def get_data_handler(data_type):
    if data_type == "fp16":
        return {
            "load": load_fp16,
            "torch_dtype": torch.float16,
            "to_torch": lambda x: torch.from_numpy(x),
            "to_kernel": lambda x: np.frombuffer(x.tobytes(), dtype=np.uint16),
            "from_kernel": lambda x, shape: np.frombuffer(x.tobytes(), dtype=np.float16).reshape(shape),
            "atol": 1e-3,
            "rtol": 1e-3,
        }
    elif data_type == "bf16":
        return {
            "load": load_bf16,
            "torch_dtype": torch.bfloat16,
            "to_torch": lambda x: torch.from_numpy(x.astype(np.float32)),
            "to_kernel": lambda x: x.flatten(),
            # --- FIX: Reshape the output from the kernel ---
            "from_kernel": lambda x, shape: x.astype(np.float32).reshape(shape),
            "atol": 0,
            "rtol": 0,
        }
    else:
        raise ValueError(f"Unknown data type: {data_type}")

# --- Test Execution ---

results = []

def discover_and_run_tests(kernel_filter=None, test_filter=None):
    """
    Discover and run test cases.
    """
    global results
    results.clear()

    print(f"Discovering tests (kernel='{kernel_filter}', test='{test_filter}')\n")

    for kernel_name in sorted(os.listdir(KERNELS_DIR)):
        if kernel_filter and kernel_name != kernel_filter:
            continue

        kernel_path = os.path.join(KERNELS_DIR, kernel_name)
        if not os.path.isdir(kernel_path):
            continue

        tests_root = os.path.join(kernel_path, "tests")
        if not os.path.exists(tests_root):
            continue

        for test_name in sorted(os.listdir(tests_root)):
            if test_filter and test_name != test_filter:
                continue

            test_path = os.path.join(tests_root, test_name)
            if not os.path.isdir(test_path):
                continue

            test_script = os.path.join(test_path, "test.py")
            if not os.path.exists(test_script):
                continue

            run_test_case(kernel_name, test_name, test_path, args.full_diff)

    return results


def run_test_case(kernel_name, test_name, test_path, full_diff):
    try:
        # --- Setup ---
        operator_name, data_type = kernel_name.rsplit('_', 1)
        operator = OPERATOR_REGISTRY[operator_name]
        data_handler = get_data_handler(data_type)

        # --- Load Test Case ---
        spec = importlib.util.spec_from_file_location(f"{kernel_name}.{test_name}", os.path.join(test_path, "test.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        case = module.generate_case()
        
        print(f"Running: {kernel_name} / {test_name} [{case['shape']}]")

        if operator_name == "conv2d":
            # Conv2D
            params = case.get("params", {})
            inputs_np = {name: data_handler["load"](os.path.join(test_path, f"{name}.bin"), tensor.shape)
                         for name, tensor in case["inputs"].items()}
            expected = data_handler["load"](os.path.join(test_path, "expected_output.bin"), case["expected"].shape)
            inputs_pt = {name: data_handler["to_torch"](tensor).to('cuda').to(data_handler["torch_dtype"])
                         for name, tensor in inputs_np.items()}
            inputs_kernel = {name: data_handler["to_kernel"](tensor) for name, tensor in inputs_np.items()}

            # PyTorch Baseline
            torch.cuda.synchronize()
            start = time.time()
            ref_out = operator.get_pytorch_baseline(inputs_pt, params)
            torch.cuda.synchronize()
            torch_time = time.time() - start

            # Kernel Execution
            torch.cuda.synchronize()
            start = time.time()
            actual_kernel_out = operator.get_kernel_call(kernel_lib, inputs_kernel, params, kernel_name)
            torch.cuda.synchronize()
            kernel_time = time.time() - start

        elif operator_name == "matmul":
            expected_shape = case["expected"].shape
            expected = data_handler["load"](os.path.join(test_path, "expected_output.bin"), expected_shape)

            if len(case["shape"]) == 3:
                m, n, k = case["shape"]
            else:
                m, k = case["shape"]
                n = expected_shape[1]

            shape_a = (m, k)
            shape_b = (k, n)

            a = data_handler["load"](os.path.join(test_path, "A.bin"), shape_a)
            b = data_handler["load"](os.path.join(test_path, "B.bin"), shape_b)


            # PyTorch Baseline
            a_pt = data_handler["to_torch"](a).to('cuda').to(data_handler["torch_dtype"])
            b_pt = data_handler["to_torch"](b).to('cuda').to(data_handler["torch_dtype"])
            torch.cuda.synchronize()
            start = time.time()
            ref_out = operator.get_pytorch_baseline(a_pt, b_pt)
            torch.cuda.synchronize()
            torch_time = time.time() - start

            # Kernel Execution
            a_kernel = data_handler["to_kernel"](a)
            b_kernel = data_handler["to_kernel"](b)
            torch.cuda.synchronize()
            start = time.time()
            actual_kernel_out = operator.get_kernel_call(kernel_lib, a_kernel, b_kernel, shape_a, expected_shape, kernel_name)
            torch.cuda.synchronize()
            kernel_time = time.time() - start

        elif operator_name == "eltw_add":
            # EltwiseAdd and MatMul
            shape = case["shape"]
            expected_shape = case["expected"].shape
            a = data_handler["load"](os.path.join(test_path, "A.bin"), shape)
            b = data_handler["load"](os.path.join(test_path, "B.bin"), shape)
            expected = data_handler["load"](os.path.join(test_path, "expected_output.bin"), expected_shape)

            # PyTorch Baseline
            a_pt = data_handler["to_torch"](a).to('cuda').to(data_handler["torch_dtype"])
            b_pt = data_handler["to_torch"](b).to('cuda').to(data_handler["torch_dtype"])
            torch.cuda.synchronize()
            start = time.time()
            ref_out = operator.get_pytorch_baseline(a_pt, b_pt)
            torch.cuda.synchronize()
            torch_time = time.time() - start

            # Kernel Execution
            a_kernel = data_handler["to_kernel"](a)
            b_kernel = data_handler["to_kernel"](b)
            torch.cuda.synchronize()
            start = time.time()
            actual_kernel_out = operator.get_kernel_call(kernel_lib, a_kernel, b_kernel, shape, expected_shape, kernel_name)
            torch.cuda.synchronize()
            kernel_time = time.time() - start

        # --- Compare ---
        # In python/run.py, replace the final "Compare" section in run_test_case

        # --- Compare and Report (common logic for all operators) ---
        actual = data_handler["from_kernel"](actual_kernel_out, case["expected"].shape)
        passed = np.allclose(actual, expected, rtol=data_handler["rtol"], atol=data_handler["atol"])
        status = "Pass"

        if not passed:
            status = "Fail"
            analyzer = ErrorAnalyzer(actual, expected, test_name, kernel_name, test_path, data_handler,  actual_u16_raw=actual_kernel_out)

            # Generate enhanced error analysis
            visualize_enhanced_errors(actual, expected, kernel_name, test_name, data_handler, actual_u16_raw=actual_kernel_out)
            
            # Generate error mask binary (as before)
            error_mask = analyzer.mismatch_mask.astype(np.uint8)
            if error_mask.ndim > 2:
                error_mask_flat = error_mask.reshape(-1, error_mask.shape[-1])
            else:
                error_mask_flat = error_mask

            mask_file_path = os.path.join(test_path, "error_mask.bin")
            with open(mask_file_path, "wb") as f:
                f.write(struct.pack("<II", error_mask_flat.shape[0], error_mask_flat.shape[1]))
                f.write(error_mask_flat.tobytes())
            print(f"  -> Error mask saved to {mask_file_path}")

            # Plot mask with enhanced hover info
            visualize_tensor_errors(actual, expected, kernel_name, test_name, args=args, data_handler=data_handler, actual_u16_raw=actual_kernel_out)

            # Generate detailed text report
            analyzer.generate_enhanced_report()
            analyzer.generate_report(full_diff=full_diff)

            # Update metrics for results table
            mismatch_count = analyzer.mismatch_count
            max_error = np.max(analyzer.diff)
            mean_absolute_percent_error = np.mean(analyzer.diff) * 100

        else:
            max_error = np.max(np.abs(actual - expected)) if actual.size > 0 else 0.0
            mean_absolute_percent_error = np.mean(np.abs(actual - expected)) * 100 if actual.size > 0 else 0.0

        results.append({
            "Kernel": kernel_name, "Test": test_name, "Shape": str(case['shape']),
            "PyTorch Time (ms)": f"{torch_time * 1000:.3f}",
            "Kernel Time (ms)": f"{kernel_time * 1000:.3f}",
            "Max Error": f"{max_error:.6f}",
            "Mean Abs Error%": f"{mean_absolute_percent_error:.6f}",
            "Status": status
        })

    except Exception as e:
        results.append({
            "Kernel": kernel_name,
            "Test": test_name,
            "Shape": "N/A",
            "PyTorch Time (ms)": "–",
            "Kernel Time (ms)": "–",
            "Max Error": "–",
            "Status": "Error",
            "Error": str(e)[:100]
        })

# --- Main ---

def print_results(results):
    if not results:
        print("No test cases were run.")
        return

    print("\n Test Results")
    print(tabulate(results, headers="keys", tablefmt="grid"))

    passed = sum(1 for r in results if "Pass" in r["Status"])
    failed = len(results) - passed
    print(f"\n{passed} passed |{failed} failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run kernel tests")
    parser.add_argument("path", nargs='?', default=None, help="Optional path to a specific kernel or test to run.")
    parser.add_argument("--k", "--kernel", dest="kernel_filter", help="Filter by kernel name (e.g., eltw_add_fp16)")
    parser.add_argument("--t", "--test", dest="test_filter", help="Filter by test case name (e.g., test_1)")
    parser.add_argument("--bld", "--build_dir", dest="build_dir", default="../build", help="Path to the build directory containing kernel_lib")
    parser.add_argument("--full-diff", action="store_true", help="Dump ALL mismatches to a detailed text report.")
    
    args = parser.parse_args()
    kernel_filter = args.kernel_filter
    test_filter = args.test_filter

    if args.path and not (kernel_filter or test_filter):
        path = os.path.normpath(args.path)
        abs_path = os.path.abspath(path)
        
        if KERNELS_DIR in abs_path:
            relative_path = os.path.relpath(abs_path, KERNELS_DIR)
            parts = relative_path.split(os.sep)
            
            if len(parts) > 0:
                kernel_filter = parts[0]
            if len(parts) > 2 and parts[1] == "tests":
                test_filter = parts[2]

    if kernel_filter:
        kernel_filter = os.path.basename(os.path.normpath(kernel_filter))
    if test_filter:
        test_filter = os.path.basename(os.path.normpath(test_filter))

    build_bindings_path = os.path.join(os.path.abspath(args.build_dir), "bindings")
    sys.path.insert(0, build_bindings_path)

    try:
        from bindings import kernel_lib
    except ImportError as e:
        print(f"Failed to import kernel_lib from {build_bindings_path}")
        print(e)
        sys.exit(1)

    results = discover_and_run_tests(kernel_filter, test_filter)
    print_results(results)

    if any("Fail" in r["Status"] or "Error" in r["Status"] for r in results):
        sys.exit(1)