import torch
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
    # FIXED: classes=1 for the iterative single-mask approach
    model = smp.Unet(
        encoder_name="resnet50",   
        encoder_weights=None,      
        in_channels=1,             
        classes=1                  
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load("diffraction_unet.pth", map_location=device, weights_only=True))
    model = model.to(device)
    
    # CRITICAL: Put the model in evaluation mode
    model.eval()

    # 3. Fetch a brand new, unseen test sample
    dataset = DiffractionDataset(sim_size=1024, target_size=512, epoch_size=1)
    
    # Extract the first (and only) sample. 
    inputs_tensor, targets_tensor = dataset[0]
    
    # Apply the exact same normalization used during training
    inputs_tensor = inputs_tensor.float()
    inputs_tensor = (inputs_tensor - inputs_tensor.min()) / (inputs_tensor.max() - inputs_tensor.min() + 1e-8)
    
    # Clone the input tensor so we can iteratively subtract from it
    current_input = inputs_tensor.clone()

    # Setup the plot: 3 rows (Rounds 1-3), 3 columns (Input, Prediction, Remainder)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle("Iterative Diffraction Splitting (Winner-Takes-All)", fontsize=16)

    # 4. Iterative Inference Loop
    for round_idx in range(3):
        # Add batch dimension: (1, 1, H, W)
        inputs_batch = current_input.unsqueeze(0).to(device)

        with torch.no_grad():
            raw_outputs = model(inputs_batch)
            probabilities = torch.sigmoid(raw_outputs)
            
            # Keep the thresholded mask on the device for tensor math
            prediction_mask_device = (probabilities > 0.7).float()

        # Extract copies to the CPU for matplotlib
        img_before_subtraction = current_input.squeeze(0).cpu().numpy()
        
        # prediction_mask_device is [1, 1, H, W], we need [H, W] for plotting
        pred_mask_np = prediction_mask_device.squeeze(0).squeeze(0).cpu().numpy()

        # --- FIXED SUBTRACTION STEP ---
        # Input image: Background is ~1.0 (white), spots are ~0.0 (black).
        # Prediction mask: Background is 0.0 (black), spots are 1.0 (white).
        # To erase a spot, we add the mask to turn the 0.0 pixel into 1.0 (background).
        current_input = torch.clamp(current_input + prediction_mask_device.squeeze(0), 0.0, 1.0)
        
        img_after_subtraction = current_input.squeeze(0).cpu().numpy()

        # 5. Plotting this round's results
        axes[round_idx, 0].imshow(img_before_subtraction, cmap='gray')
        axes[round_idx, 0].set_title(f"Round {round_idx + 1} - Input Image")
        
        axes[round_idx, 1].imshow(pred_mask_np, cmap='gray')
        axes[round_idx, 1].set_title(f"Round {round_idx + 1} - Extracted Pattern")
        
        axes[round_idx, 2].imshow(img_after_subtraction, cmap='gray')
        axes[round_idx, 2].set_title(f"Round {round_idx + 1} - Image Remainder")

    # Clean up axes
    for ax in axes.flatten():
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test()
