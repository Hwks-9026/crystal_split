import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

# Import the dataset we built
from dataset import DiffractionDataset

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
        encoder_name="resnet50",   
        encoder_weights=None,      
        in_channels=1,             
        classes=1                  
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("diffraction_unet.pth", map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    # 3. Fetch a brand new, unseen test sample
    dataset = DiffractionDataset(sim_size=1024, target_size=512, epoch_size=1)
    inputs_tensor, targets_tensor = dataset[0]
    
    # Apply normalization
    inputs_tensor = inputs_tensor.float()
    inputs_tensor = (inputs_tensor - inputs_tensor.min()) / (inputs_tensor.max() - inputs_tensor.min() + 1e-8)
    
    # Clone the input tensor for both the model's iteration and our ideal iteration
    current_input = inputs_tensor.clone()
    ideal_current_input = inputs_tensor.clone().cpu().squeeze().numpy()

    # Setup the plot: 3 rows, 5 columns
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))
    fig.suptitle("Iterative Diffraction Splitting (Winner-Takes-All)", fontsize=16)

    # 4. Iterative Inference Loop
    for round_idx in range(3):
        # Add batch dimension: (1, 1, H, W)
        inputs_batch = current_input.unsqueeze(0).to(device)

        with torch.no_grad():
            raw_outputs = model(inputs_batch)
            probabilities = torch.sigmoid(raw_outputs)
            prediction_mask_device = (probabilities > 0.7).float()

        # Extract copies to the CPU for matplotlib
        img_before_subtraction = current_input.squeeze(0).cpu().numpy()
        pred_mask_np = prediction_mask_device.squeeze(0).squeeze(0).cpu().numpy()
        desired_mask_np = targets_tensor[round_idx].squeeze().cpu().numpy()

        # --- MODEL SUBTRACTION STEP ---
        current_input = torch.clamp(current_input + prediction_mask_device.squeeze(0), 0.0, 1.0)
        img_after_subtraction = current_input.squeeze(0).cpu().numpy()

        # --- IDEAL SUBTRACTION STEP ---
        # Subtracting the ground truth from our tracked ideal input
        ideal_current_input = np.clip(ideal_current_input + desired_mask_np, 0.0, 1.0)
        ideal_remainder_np = ideal_current_input.copy()

        # 5. Plotting this round's results
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
    plt.show()

if __name__ == "__main__":
    test()
