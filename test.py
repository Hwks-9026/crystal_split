import argparse
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from skimage.feature import peak_local_max
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

import fabio
from dataset import DiffractionDataset


def compute_peak_metrics(pred_coords, true_coords, tolerance=3.0):
    """
    Calculates Precision, Recall, and F1 score for peak detection.
    A true positive is a predicted peak within distance of a true peak.
    """
    if len(pred_coords) == 0 or len(true_coords) == 0:
        return 0.0, 0.0, 0.0
        
    tree = cKDTree(true_coords)
    distances, indices = tree.query(pred_coords)
    
    valid_matches = distances <= tolerance
    tp = np.sum(valid_matches)
    fp = len(pred_coords) - tp
    
    matched_true_indices = np.unique(indices[valid_matches])
    fn = len(true_coords) - len(matched_true_indices)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def isolate_and_find_peaks(raw_image_inverted, unet_predicted_mask, threshold=0.5):
    """
    Isolates diffraction spots by masking the raw image with the AI's probability mask,
    then executes a classical peak finding algorithm on the cleaned output.
    """
    binary_mask = (unet_predicted_mask > threshold).astype(float)
    cleaned_image = np.where(binary_mask > 0, raw_image_inverted, 1.0)
    
    search_cleaned = 1.0 - cleaned_image
    hybrid_coords = peak_local_max(search_cleaned, min_distance=3, threshold_abs=0.05)
    
    return hybrid_coords, cleaned_image


def load_and_preprocess_cbf(cbf_path):
    """
    Loads a CBF file, handles dead pixels, normalizes, inverts to match dataset expectations,
    and pads dimensions to be divisible by 32 for the UNet.
    """
    if fabio is None:
        raise ImportError("The 'fabio' library is required to load .cbf files. Run: pip install fabio")
        
    cbf_img = fabio.open(cbf_path)
    data = cbf_img.data.astype(np.float32)
    
    # 1. Dead pixel filtering
    dead_pixel_mask = data < 0
    if dead_pixel_mask.any():
        filtered_data = median_filter(data, size=3)
        data[dead_pixel_mask] = filtered_data[dead_pixel_mask]
        
    # 2. Robust Normalization
    # Ignore extreme hot pixels/direct beam by clipping to the 99.9th percentile
    vmin = np.percentile(data, 1.0)
    vmax = np.percentile(data, 99.9)
    
    data = np.clip(data, vmin, vmax)
    data = (data - vmin) / (vmax - vmin + 1e-8)
    
    # Invert to match the UNet's expected domain (dark peaks on a bright background)
    data = 1.0 - data
    
    # 3. Pad to nearest multiple of 32 for UNet compatibility
    h, w = data.shape
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    if pad_h > 0 or pad_w > 0:
        data = np.pad(data, ((0, pad_h), (0, pad_w)), mode='reflect')
        
    return torch.tensor(data).unsqueeze(0).unsqueeze(0)

def test(cbf_path=None):
    """
    Loads the trained model, performs iterative lattice extraction, isolates peaks,
    computes quantitative metrics per lattice, and renders with matplotlib.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing on device: {device}")

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=2, classes=1)
    
    try:
        model.load_state_dict(torch.load("diffraction_unet_trev.pth", map_location=device, weights_only=True))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print("Warning: 'diffraction_unet_trev.pth' not found. Using untrained weights for demonstration.")
        
    model = model.to(device).eval()

    is_real_data = cbf_path is not None
    if is_real_data:
        print(f"Loading real CBF data from: {cbf_path}")
        inputs_tensor = load_and_preprocess_cbf(cbf_path)
        targets_tensor = None
        metrics_text = f"Source: {cbf_path.split('/')[-1]}\nMode: Inference Only (No GT)\n\n"
    else:
        if DiffractionDataset is None:
            raise ImportError("DiffractionDataset not found and no --cbf argument provided.")
        print("Loading simulation data from DiffractionDataset...")
        dataset = DiffractionDataset(sim_size=1024, target_size=512, epoch_size=1)
        inputs_tensor, targets_tensor, metadata = dataset[0]
        inputs_tensor = inputs_tensor.float().unsqueeze(0)
        inputs_tensor = (inputs_tensor - inputs_tensor.min()) / (inputs_tensor.max() - inputs_tensor.min() + 1e-8)
        metrics_text = f"Simulation Metadata:\nCamera Length: {metadata['camera_length']:.1f}mm\nWavelength: {metadata['wavelength']}Å\n\n"

    original_raw_image = inputs_tensor.squeeze().cpu().numpy().copy()
    total_predicted_mask = np.zeros_like(original_raw_image)
    
    current_input = inputs_tensor.clone().to(device).squeeze(0)
    ideal_current_input = inputs_tensor.clone().cpu().squeeze().numpy()

    fig = plt.figure(figsize=(25, 20))
    title = "Iterative Diffraction Splitting & Peak Finding"
    title += f" ({'CBF Inference' if is_real_data else 'Simulation'})"
    fig.suptitle(title, fontsize=20, weight='bold')
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.5])

    num_rounds = 2

    for round_idx in range(num_rounds):
        B, C, H, W = current_input.unsqueeze(0).shape
        input_flat = current_input.view(1, -1)
        min_vals, min_indices = input_flat.min(dim=1)
        
        seed_flat = torch.zeros_like(input_flat)
        if min_vals[0] < 0.8:
            seed_flat.scatter_(1, min_indices.unsqueeze(1), 1.0)

        seed_channel = F.max_pool2d(seed_flat.view(1, 1, H, W).to(device), kernel_size=31, stride=1, padding=15)
        inputs_batch = torch.cat([current_input.unsqueeze(0), seed_channel], dim=1)

        with torch.no_grad():
            probabilities = torch.sigmoid(model(inputs_batch))
            prediction_mask_device = (probabilities > 0.7).float()

        img_before_subtraction = current_input.squeeze(0).cpu().numpy()
        pred_mask_np = prediction_mask_device.squeeze(0).squeeze(0).cpu().numpy()
        total_predicted_mask = np.maximum(total_predicted_mask, pred_mask_np)
        
        current_input = torch.clamp(current_input + prediction_mask_device.squeeze(0), 0.0, 1.0)
        img_after_subtraction = current_input.squeeze(0).cpu().numpy()

        ax0 = fig.add_subplot(gs[round_idx, 0]); ax0.imshow(img_before_subtraction, cmap='gray'); ax0.set_title(f"R{round_idx + 1}: Input"); ax0.axis("off")
        ax2 = fig.add_subplot(gs[round_idx, 2]); ax2.imshow(pred_mask_np, cmap='gray'); ax2.set_title(f"R{round_idx + 1}: Predicted Mask"); ax2.axis("off")
        ax3 = fig.add_subplot(gs[round_idx, 3]); ax3.imshow(img_after_subtraction, cmap='gray'); ax3.set_title(f"R{round_idx + 1}: Model Remainder"); ax3.axis("off")

        if not is_real_data:
            desired_mask_np = targets_tensor[round_idx].squeeze().cpu().numpy()
            ideal_current_input = np.clip(ideal_current_input + desired_mask_np, 0.0, 1.0)
            ideal_remainder_np = ideal_current_input.copy()
            
            ax1 = fig.add_subplot(gs[round_idx, 1]); ax1.imshow(desired_mask_np, cmap='gray'); ax1.set_title(f"R{round_idx + 1}: GT Mask"); ax1.axis("off")
            ax4 = fig.add_subplot(gs[round_idx, 4]); ax4.imshow(ideal_remainder_np, cmap='gray'); ax4.set_title(f"R{round_idx + 1}: Ideal Remainder"); ax4.axis("off")
            
            true_coords = peak_local_max(desired_mask_np, min_distance=3, threshold_abs=0.5)
            
            if len(true_coords) == 0:
                metrics_text += f"Lattice {round_idx + 1}: No ground truth peaks found.\n"
            else:
                round_cleaned_image = np.where(pred_mask_np > 0.5, original_raw_image, 1.0)
                hybrid_coords_round = peak_local_max(1.0 - round_cleaned_image, min_distance=3, threshold_abs=0.05)
                ai_p, ai_r, ai_f1 = compute_peak_metrics(hybrid_coords_round, true_coords)
                metrics_text += f"Lattice {round_idx + 1} ({len(true_coords)} true peaks):\n"
                metrics_text += f"Precision: {ai_p:.2f} | Recall: {ai_r:.2f} | F1: {ai_f1:.2f}\n\n"
        else:
            blank = np.zeros_like(img_before_subtraction)
            ax1 = fig.add_subplot(gs[round_idx, 1]); ax1.imshow(blank, cmap='gray'); ax1.set_title(f"R{round_idx + 1}: GT Mask (N/A)"); ax1.axis("off")
            ax4 = fig.add_subplot(gs[round_idx, 4]); ax4.imshow(blank, cmap='gray'); ax4.set_title(f"R{round_idx + 1}: Ideal Remainder (N/A)"); ax4.axis("off")
            
            round_cleaned_image = np.where(pred_mask_np > 0.5, original_raw_image, 1.0)
            hybrid_coords_round = peak_local_max(1.0 - round_cleaned_image, min_distance=3, threshold_abs=0.05)
            metrics_text += f"Lattice {round_idx + 1}: Extracted {len(hybrid_coords_round)} peaks (GT unavailable)\n"

    hybrid_coords_total, cleaned_img_total = isolate_and_find_peaks(original_raw_image, total_predicted_mask, threshold=0.5)

    ax_orig = fig.add_subplot(gs[2, 0:2])
    ax_orig.imshow(original_raw_image, cmap='gray')
    ax_orig.set_title(f"Original Raw Image {'(Padded)' if is_real_data else ''}")
    ax_orig.axis("off")

    ax_clean = fig.add_subplot(gs[2, 2:4])
    ax_clean.imshow(cleaned_img_total, cmap='gray')
    if len(hybrid_coords_total) > 0:
        ax_clean.plot(hybrid_coords_total[:, 1], hybrid_coords_total[:, 0], 'g.', markersize=10, alpha=0.8)
    ax_clean.set_title(f"AI Masked Image + Peak Finder ({len(hybrid_coords_total)} total peaks)")
    ax_clean.axis("off")

    ax_metrics = fig.add_subplot(gs[2, 4])
    ax_metrics.axis("off")
    ax_metrics.text(0.05, 0.95, "Quantitative Peak Analysis\n" + "-"*30 + "\n" + metrics_text, 
                    transform=ax_metrics.transAxes, fontsize=14, verticalalignment='top', family='monospace')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test UNet for Crystal Lattice Separation.")
    parser.add_argument("--cbf", type=str, default=None, help="Path to a .cbf file for inference. If omitted, uses simulation dataset.")
    args = parser.parse_args()
    
    test(cbf_path=args.cbf)
