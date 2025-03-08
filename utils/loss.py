import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1, temperature=None, margin=1.0, dynamic_beta=False, beta_min=0.1, beta_max=1.0, beta_decay=0.99, class_aware_beta=False):
        """
        Contrastive Loss with dynamic and class-aware divergence penalties.

        Args:
            temp (float): Default temperature parameter.
            temperature (float): Overrides temp if provided.
            margin (float): Margin for contrastive loss.
            dynamic_beta (bool): Whether to use dynamic beta.
            beta_min (float): Minimum value for dynamic beta.
            beta_max (float): Maximum value for dynamic beta.
            beta_decay (float): Decay rate for dynamic beta.
            class_aware_beta (bool): Whether to use class-aware divergence penalty.
        """
        super(ContrastiveLoss, self).__init__()

        # Handle temperature parameter
        self.temp = temperature if temperature is not None else temp

        self.margin = margin
        self.dynamic_beta = dynamic_beta
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_decay = beta_decay
        self.class_aware_beta = class_aware_beta
        self.beta = beta_max  # Initialize beta to its maximum value

    def update_beta(self, epoch):
        """
        Update the dynamic beta based on the training progress.

        Args:
            epoch (int): Current epoch.
        """
        if self.dynamic_beta:
            self.beta = self.beta_min + (self.beta_max - self.beta_min) * (self.beta_decay ** epoch)

    def forward(self, z_prev, z_present, z_serv, labels=None, epoch=None):
        if epoch is not None:
            self.update_beta(epoch)  # Update beta based on the current epoch
        
        # Print tensor shapes for debugging
        print(f"Input shapes: z_prev={z_prev.shape}, z_present={z_present.shape}, z_serv={z_serv.shape}")
        
        # Instead of using transpose directly, we'll reshape the tensors to ensure proper dimensions
        batch_size = z_prev.size(0)
        
        # Reshape features to 2D if they're not already
        if len(z_prev.shape) > 2:
            z_prev = z_prev.reshape(batch_size, -1)
        if len(z_present.shape) > 2:
            z_present = z_present.reshape(batch_size, -1)
        if len(z_serv.shape) > 2:
            z_serv = z_serv.reshape(batch_size, -1)
            
        print(f"Reshaped: z_prev={z_prev.shape}, z_present={z_present.shape}, z_serv={z_serv.shape}")
        
        # Normalize features
        z_prev = F.normalize(z_prev, p=2, dim=1)
        z_present = F.normalize(z_present, p=2, dim=1)
        z_serv = F.normalize(z_serv, p=2, dim=1)
        
        # Compute similarity matrices correctly
        # Use permute or transpose correctly based on the tensor dimensions
        sim_prev_present = torch.matmul(z_prev, z_present.permute(1, 0)) / self.temp
        sim_prev_serv = torch.matmul(z_prev, z_serv.permute(1, 0)) / self.temp
        
        print(f"Similarity matrix shapes: sim_prev_present={sim_prev_present.shape}, sim_prev_serv={sim_prev_serv.shape}")
        
        # Compute contrastive loss
        logits = torch.cat([sim_prev_present, sim_prev_serv], dim=1)
        contrastive_labels = torch.arange(batch_size, device=z_prev.device)
        loss = F.cross_entropy(logits, contrastive_labels)
        
        # Apply dynamic divergence penalty
        if self.dynamic_beta:
            loss = loss * self.beta
        
        # Apply class-aware divergence penalty (if labels are provided)
        if self.class_aware_beta and labels is not None:
            # Use the provided labels for class weighting
            class_counts = torch.bincount(labels, minlength=labels.max().item() + 1)
            class_weights = 1.0 / (class_counts[labels] + 1e-6)  # Inverse frequency as weights
            weighted_loss = loss * class_weights
            loss = weighted_loss.mean()
        
        return loss
