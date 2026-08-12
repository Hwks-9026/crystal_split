import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision.transforms import v2
from torchvision import tv_tensors
from torch.utils.data import DataLoader
from dataset import DiffractionDataset

def compute_autocorrelation(x):
    """
    Computes the 2D autocorrelation of a batched image tensor using the Wiener-Khinchin theorem.
    Returns the max-normalized autocorrelation map.
    """
    B, C, H, W = x.shape
    x_padded = F.pad(x, (W//2, W//2, H//2, H//2), mode='constant', value=0)

    fft_x = torch.fft.fft2(x_padded)
    autocorr = torch.fft.fftshift(torch.fft.ifft2(torch.abs(fft_x) ** 2).real, dim=(-2, -1))

    return autocorr / (autocorr.amax(dim=(-2, -1), keepdim=True) + 1e-8)

def calc_single_target_loss(p, t, focal_alpha=0.80, gamma=2.0, autocorr_weight=3.0, tversky_alpha=0.3, tversky_beta=0.7):
    """
    Calculates loss for target prediction.
    Combines Focal Loss, Tversky Loss, Mass Error, and Autocorrelation MSE.
    """
    B = p.size(0)
    p_flat, t_flat = p.view(B, -1), t.view(B, -1)
    
    p_t = p_flat * t_flat + (1 - p_flat) * (1 - t_flat)
    focal_batch = (-focal_alpha * ((1 - p_t) ** gamma) * torch.log(p_t + 1e-8)).mean(dim=1)
    
    tp, fp, fn = (p_flat * t_flat).sum(dim=1), (p_flat * (1 - t_flat)).sum(dim=1), ((1 - p_flat) * t_flat).sum(dim=1)
    tversky_loss = 1.0 - (tp / (tp + tversky_alpha * fp + tversky_beta * fn + 1e-8))
    
    mass_error = torch.abs(p.mean(dim=[1, 2, 3]) - t.mean(dim=[1, 2, 3]))
    autocorr_loss = F.mse_loss(compute_autocorrelation(p), compute_autocorrelation(t), reduction='none').mean(dim=[1, 2, 3])
    
    return focal_batch + tversky_loss + (autocorr_weight * autocorr_loss) + mass_error

def generate_seed_channel(target):
    """
    Creates an attention seed channel by picking the brightest coordinate in the target mask,
    converting it into a 31x31 block so the CNN can see it.
    """
    B, C, H, W = target.shape
    target_flat = target.view(B, -1)
    max_vals, max_indices = target_flat.max(dim=1)
    
    seed_flat = torch.zeros_like(target_flat).scatter_(1, max_indices.unsqueeze(1), 1.0)
    seed_channel = F.max_pool2d(seed_flat.view(B, C, H, W), kernel_size=31, stride=1, padding=15)
    
    has_spots = (max_vals > 0).float().view(B, 1, 1, 1)
    return seed_channel * has_spots, has_spots.view(B)

def save_visualization_hook(epoch, batch_idx, round_idx, current_inputs, seed, target, pred, save_path="latest_training_viz.png"):
    """
    Renders visual snapshot of the training state (Useful for ensuring further training will be worthwile).
    """
    img_in = current_inputs[0, 0].cpu().detach().float().numpy()
    img_seed = seed[0, 0].cpu().detach().float().numpy()
    img_tgt = target[0, 0].cpu().detach().float().numpy()
    img_pred = pred[0, 0].cpu().detach().float().numpy()
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(img_in, cmap='magma')
    axes[0].set_title("Current Input")
    axes[1].imshow(img_seed, cmap='magma')
    axes[1].set_title("Seed (Attention)")
    axes[2].imshow(img_tgt, cmap='magma')
    axes[2].set_title(f"Target (Round {round_idx})")
    axes[3].imshow(img_pred, cmap='magma')
    axes[3].set_title("Model Prediction")
    
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    
    temp_path = "tmp_" + save_path
    plt.savefig(temp_path, dpi=100, bbox_inches='tight')
    plt.close(fig) 
    
    os.replace(temp_path, save_path)

def train():
    """
    Main execution pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=2, classes=1)
    model = model.to(device, memory_format=torch.channels_last)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    augmentations = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5)])
    
    train_dataset = DiffractionDataset(sim_size=1024, target_size=512, epoch_size=1000)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    num_epochs = 100
    
    try:
        for epoch in range(num_epochs):
            model.train() 
            running_loss = 0.0
            epoch_loss = 0.0
            for batch_idx, (inputs, targets, metadata) in enumerate(train_loader):
                inputs = inputs.to(device, memory_format=torch.channels_last).float()
                targets = targets.to(device)

                b_min = inputs.amin(dim=(2, 3), keepdim=True)
                b_max = inputs.amax(dim=(2, 3), keepdim=True)
                inputs = (inputs - b_min) / (b_max - b_min + 1e-8)

                inputs, targets = augmentations(tv_tensors.Image(inputs), tv_tensors.Mask(targets))
                inputs, targets = inputs.contiguous(), targets.contiguous()

                optimizer.zero_grad()
                current_inputs = inputs.clone()
                total_loss = 0.0
                
                for round_idx in range(2):
                    t_i = targets[:, round_idx:round_idx+1, :, :]
                    seed_channel, has_spots = generate_seed_channel(t_i)
                    
                    outputs = model(torch.cat([current_inputs, seed_channel], dim=1))
                    probs = torch.sigmoid(outputs)
                    
                    loss = calc_single_target_loss(probs, t_i)
                    total_loss += (loss * has_spots).mean()
                    
                    current_inputs = torch.clamp(current_inputs + probs.detach(), 0.0, 1.0)
                    
                    if (batch_idx + 1) % 10 == 0 and round_idx == 0:
                        save_visualization_hook(epoch, batch_idx, round_idx, current_inputs, seed_channel, t_i, probs)

                total_loss.backward()
                optimizer.step()
                running_loss += total_loss.item()
                epoch_loss += total_loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {(running_loss/10):.4f}")
                    running_loss = 0.0

            avg_epoch_loss = epoch_loss / len(train_loader)
            scheduler.step(avg_epoch_loss)
            

    except KeyboardInterrupt:
        print("\nTraining interrupted early...")
    finally:
        torch.save(model.state_dict(), "diffraction_unet_trev.pth")
        print("Model saved successfully.")

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    train()
