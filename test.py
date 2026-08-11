import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from skimage.feature import peak_local_max # Added skimage

# Import the dataset we built
from dataset import DiffractionDataset

# --- NEW HYBRID PIPELINE FUNCTIONS ---
def evaluate_hybrid_pipeline(raw_image_inverted, unet_predicted_mask, threshold=0.5):
    """
    Takes the raw diffraction image and the AI's probability mask,
    cleans the background, and finds the exact peak coordinates.
    Assumes diffraction spots are DARK (close to 0) and background is LIGHT (close to 1).
    """
    binary_mask = (unet_predicted_mask > threshold).astype(float)
    
    # Since your spots are dark (0), we mask the background with 1.0 (light)
    cleaned_image = np.where(binary_mask > 0, raw_image_inverted, 1.0)
    
    # Classic peak_local_max looks for BRIGHT spots. 
    # We invert the images mathematically just for the search algorithm.
    search_cleaned = 1.0 - cleaned_image
    search_raw = 1.0 - raw_image_inverted
    
    hybrid_coords = peak_local_max(
        search_cleaned, 
        min_distance=3,       
        threshold_abs=0.05    
    )
    
    baseline_coords = peak_local_max(
        search_raw, 
        min_distance=3, 
        threshold_abs=0.05
    )
    
    return hybrid_coords, baseline_coords, cleaned_image

def plot_comparison(raw_image, cleaned_image, baseline_coords, hybrid_coords):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
    fig.suptitle("Final Output: AI Masking + Classical Peak Finding", fontsize=16)
    
    # Plot 1: Classic Algorithm Alone
    axes[0].imshow(raw_image, cmap='gray')
    axes[0].autoscale(False)
    if len(baseline_coords) > 0:
        axes[0].plot(baseline_coords[:, 1], baseline_coords[:, 0], 'r.', markersize=8, alpha=0.7)
    axes[0].set_title(f"Classic Peak Finder Only\n({len(baseline_coords)} spots found - Lots of noise)")
    
    # Plot 2: Hybrid (AI + Classic)
    axes[1].imshow(cleaned_image, cmap='gray')
    axes[1].autoscale(False)
    if len(hybrid_coords) > 0:
        axes[1].plot(hybrid_coords[:, 1], hybrid_coords[:, 0], 'g.', markersize=8, alpha=0.7)
    axes[1].set_title(f"AI Mask + Classic Peak Finder\n({len(hybrid_coords)} spots found - Cleaned)")
    
    plt.tight_layout()
    plt.show()
# -------------------------------------


def test():
    # 1. Setup Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Testing on device: {device}")

    # 2. Recreate the Model Architecture
    model = smp.Unet(
        encoder_name="tu-convnext_base",
        encoder_weights=None,
        in_channels=2,
        classes=1
    )
    # Load the trained weights
    model.load_state_dict(torch.load("diffraction_unet_trev.pth", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 3. Fetch a brand new, unseen test sample
    dataset = DiffractionDataset(sim_size=1024, target_size=512, epoch_size=1)
    
    # Extract metadata here!
    inputs_tensor, targets_tensor, metadata = dataset[0]

    # Apply normalization
    inputs_tensor = inputs_tensor.float()
    inputs_tensor = (inputs_tensor - inputs_tensor.min()) / (inputs_tensor.max() - inputs_tensor.min() + 1e-8)

    # Save absolute original for the hybrid pipeline later
    original_raw_image = inputs_tensor.squeeze(0).cpu().numpy().copy()
    
    # Accumulator for all 3 rounds of masks (used for final 1x2 plot)
    total_predicted_mask = np.zeros_like(original_raw_image)
    
    # List to store individual masks for per-lattice quantitative analysis
    masks_per_round = []

    # Clone the input tensor for both the model's iteration and our ideal iteration
    current_input = inputs_tensor.clone().to(device)
    ideal_current_input = inputs_tensor.clone().cpu().squeeze().numpy()

    # Setup the plot: 3 rows, 5 columns (ORIGINAL VISUALIZATION)
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle("Iterative Diffraction Splitting (Winner-Takes-All)", fontsize=16)

    for round_idx in range(3):

        # 1. Generate Seed Channel from the INPUT image
        B, C, H, W = current_input.unsqueeze(0).shape
        input_flat = current_input.view(1, -1)

        # Find the darkest spot (closest to 0)
        min_vals, min_indices = input_flat.min(dim=1)
        seed_flat = torch.zeros_like(input_flat)

        # Only create a seed if a sufficiently dark spot exists
        if min_vals[0] < 0.8:
            seed_flat.scatter_(1, min_indices.unsqueeze(1), 1.0)

        seed_channel = seed_flat.view(1, 1, H, W).to(device)

        # DILATE THE SEED
        seed_channel = F.max_pool2d(seed_channel, kernel_size=31, stride=1, padding=15)

        # 2. Concatenate the input image and the seed channel
        inputs_batch = torch.cat([current_input.unsqueeze(0), seed_channel], dim=1)

        with torch.no_grad():
            raw_outputs = model(inputs_batch)
            probabilities = torch.sigmoid(raw_outputs)
            prediction_mask_device = (probabilities > 0.7).float()

        # --- MODEL SUBTRACTION STEP ---
        img_before_subtraction = current_input.squeeze(0).cpu().numpy()
        pred_mask_np = prediction_mask_device.squeeze(0).squeeze(0).cpu().numpy()
        
        # Accumulate the mask for our hybrid 1x2 plot step
        total_predicted_mask = np.maximum(total_predicted_mask, pred_mask_np)
        
        # Save this specific round's mask for quantitative analysis
        masks_per_round.append(pred_mask_np.copy())
        
        desired_mask_np = targets_tensor[round_idx].squeeze().cpu().numpy()
        current_input = torch.clamp(current_input + prediction_mask_device.squeeze(0), 0.0, 1.0)
        img_after_subtraction = current_input.squeeze(0).cpu().numpy()

        ideal_current_input = np.clip(ideal_current_input + desired_mask_np, 0.0, 1.0)
        ideal_remainder_np = ideal_current_input.copy()

        # Column 1: Input
        axes[round_idx, 0].imshow(img_before_subtraction, cmap='gray')
        axes[round_idx, 0].set_title(f"Round {round_idx + 1} - Input Image")

        # Column 2: Desired Output (Ground Truth)
        axes[round_idx, 1].imshow(desired_mask_np, cmap='gray')
        axes[round_idx, 1].set_title(f"Round {round_idx + 1} - Desired Output")

        # Column 3: Model Prediction
        axes[round_idx, 2].imshow(pred_mask_np, cmap='gray')
        axes[round_idx, 2].set_title(f"Round {round_idx + 1} - Extracted Pattern")

        # Column 4: Model Remainder
        axes[round_idx, 3].imshow(img_after_subtraction, cmap='gray')
        axes[round_idx, 3].set_title(f"Round {round_idx + 1} - Model Remainder")

        # Column 5: Ideal Remainder
        axes[round_idx, 4].imshow(ideal_remainder_np, cmap='gray')
        axes[round_idx, 4].set_title(f"Round {round_idx + 1} - Ideal Remainder")

    # Clean up axes
    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.show() # Renders the 3x5 grid first

    # --- DETAILED ANALYSIS PER ROUND ---
    print("\n--- Quantitative Peak Analysis ---")
    print(f"Simulation Metadata: Camera Length={metadata['camera_length']:.1f}mm, Wavelength={metadata['wavelength']}Å\n")
    
    for round_idx in range(3):
        desired_mask = targets_tensor[round_idx].squeeze().cpu().numpy()
        
        # Find ground truth peaks for THIS specific lattice
        true_coords = peak_local_max(desired_mask, min_distance=3, threshold_abs=0.5)
        
        if len(true_coords) == 0:
            print(f"Lattice {round_idx + 1}: No ground truth peaks (empty fragment).")
            continue 
            
        current_pred_mask = masks_per_round[round_idx]
        
        # Mask the raw image using ONLY this round's predicted mask
        cleaned_image = np.where(current_pred_mask > 0.5, original_raw_image, 1.0)
        hybrid_coords = peak_local_max(1.0 - cleaned_image, min_distance=3, threshold_abs=0.05)
        
        # Calculate Classical baseline on the RAW image
        baseline_coords = peak_local_max(1.0 - original_raw_image, min_distance=3, threshold_abs=0.05)
        
        # Compare metrics
        ai_p, ai_r, ai_f1 = compute_peak_metrics(hybrid_coords, true_coords)
        base_p, base_r, base_f1 = compute_peak_metrics(baseline_coords, true_coords)
        
        print(f"Lattice {round_idx + 1} ({len(true_coords)} true peaks):")
        print(f"  AI Hybrid -> Precision: {ai_p:.2f} | Recall: {ai_r:.2f} | F1: {ai_f1:.2f}")
        print(f"  Classical -> Precision: {base_p:.2f} | Recall: {base_r:.2f} | F1: {base_f1:.2f}")

    # --- RUN THE HYBRID PIPELINE AFTER THE ITERATIVE LOOP (ORIGINAL FINAL PLOT) ---
    print("\nRunning Classical Peak Finder on Combined AI Masked Result...")
    hybrid_coords, baseline_coords, cleaned_img = evaluate_hybrid_pipeline(
        original_raw_image, 
        total_predicted_mask,
        threshold=0.5
    )
    
    plot_comparison(original_raw_image, cleaned_img, baseline_coords, hybrid_coords)

from scipy.spatial import cKDTree

def compute_peak_metrics(pred_coords, true_coords, tolerance=3.0):
    """Calculates Precision, Recall, and F1 score using KD-Tree matching."""
    if len(pred_coords) == 0 or len(true_coords) == 0:
        return 0.0, 0.0, 0.0
        
    tree = cKDTree(true_coords)
    distances, indices = tree.query(pred_coords)
    
    # A true positive is a predicted peak within `tolerance` of a true peak
    valid_matches = distances <= tolerance
    tp = np.sum(valid_matches)
    fp = len(pred_coords) - tp
    
    # To find false negatives, we count how many unique true peaks were matched.
    # If a true peak wasn't matched by anything, it's a missed spot.
    matched_true_indices = np.unique(indices[valid_matches])
    fn = len(true_coords) - len(matched_true_indices)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


if __name__ == "__main__":
    test()


