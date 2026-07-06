import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torchvision.transforms import v2
from torchvision import tv_tensors
from dataset import DiffractionDataset
import torch.nn.functional as F
import sps

def compute_autocorrelation(x):
    #2d autocorrelation using the Wiener-Khinchin theorem
    B, C, H, W = x.shape
    
    x_padded = F.pad(x, (W//2, W//2, H//2, H//2), mode='constant', value=0)
    
    fft_x = torch.fft.fft2(x_padded)
    
    power_spectrum = torch.abs(fft_x) ** 2
    
    autocorr = torch.fft.ifft2(power_spectrum).real
    
    autocorr = torch.fft.fftshift(autocorr, dim=(-2, -1))
    
    autocorr_max = autocorr.amax(dim=(-2, -1), keepdim=True) + 1e-8
    autocorr_normalized = autocorr / autocorr_max
    
    return autocorr_normalized


def train():
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    print(f"Training on device: {device}")

    model = smp.Unet(
        encoder_name="resnet50",   
        encoder_weights=None,      
        in_channels=1,             
        classes=1
    ).to(device)
    model = model.to(device, memory_format=torch.channels_last)

    optimizer = sps.Sps(model.parameters())

    augmentations = v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
    ])

    train_dataset = DiffractionDataset(sim_size=1024, target_size=512, epoch_size=1000)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=22)

    # --- DEFINE IT HERE, BEFORE THE EPOCH LOOP ---
    def calc_single_target_loss(p, t, alpha=0.80, gamma=2.0, autocorr_weight=3.0):
        batch_size = p.size(0)
        
        # Flatten to [B, H*W]
        p_flat = p.view(batch_size, -1)
        t_flat = t.view(batch_size, -1)
        
        # Focal Loss
        p_t = p_flat * t_flat + (1 - p_flat) * (1 - t_flat)
        focal = -alpha * ((1 - p_t) ** gamma) * torch.log(p_t + 1e-8)
        focal_batch = focal.mean(dim=1) 
        
        # Dice Loss
        intersection = (p_flat * t_flat).sum(dim=1)
        union = p_flat.sum(dim=1) + t_flat.sum(dim=1)
        dice_batch = 1.0 - (2.0 * intersection + 1e-8) / (union + 1e-8) 
        
        # Mass error
        pred_mass = p.mean(dim=[1, 2, 3]) 
        tgt_mass = t.mean(dim=[1, 2, 3])
        mass_error = torch.abs(pred_mass - tgt_mass)
        
        # Autocorrelation Loss
        p_autocorr = compute_autocorrelation(p)
        t_autocorr = compute_autocorrelation(t)
        autocorr_loss = F.mse_loss(p_autocorr, t_autocorr, reduction='none').mean(dim=[1, 2, 3])
        
        return focal_batch + dice_batch + (autocorr_weight * autocorr_loss) + (5.0 * mass_error)

    num_epochs = 20
    
    try:
        for epoch in range(num_epochs):
            model.train() 
            running_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device, memory_format=torch.channels_last), targets.to(device)
                
                inputs = inputs.float()
                inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min() + 1e-8)
                
                inputs = tv_tensors.Image(inputs)
                targets = tv_tensors.Mask(targets)
                inputs, targets = augmentations(inputs, targets)
                
                inputs = inputs.contiguous()
                targets = targets.contiguous()

                optimizer.zero_grad()

                current_inputs = inputs.clone()
                total_loss = 0.0
                
                # Track which targets have already been "claimed" by the model
                valid_targets_mask = (targets.amax(dim=(2, 3)) > 0).float() 

                # Iterative Training Loop
                for round_idx in range(3):
                    outputs = model(current_inputs)
                    probs = torch.sigmoid(outputs)
                    
                    channel_losses = []
                    for i in range(3):
                        t_i = targets[:, i:i+1, :, :] 
                        
                        # NOW THIS WILL WORK
                        loss_i = calc_single_target_loss(probs, t_i) 
                        
                        is_available = valid_targets_mask[:, i]
                        loss_i = loss_i * is_available + 1e5 * (1.0 - is_available)
                        
                        channel_losses.append(loss_i.unsqueeze(1))
                        
                    channel_losses = torch.cat(channel_losses, dim=1)
                    
                    min_loss, best_target_indices = torch.min(channel_losses, dim=1)
                    total_loss += min_loss.mean()
                    
                    valid_targets_mask.scatter_(1, best_target_indices.unsqueeze(1), 0.0)
                    
                    # Soft Subtraction
                    current_inputs = torch.clamp(current_inputs + probs, 0.0, 1.0)

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {(running_loss/10):.4f}")
                    running_loss = 0.0

    except KeyboardInterrupt:
        print("\nTraining interrupted early...")
    finally:
        torch.save(model.state_dict(), "diffraction_unet.pth")
        print("Model saved successfully.")

# ... (rest of the script) ...

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    train()
