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
        
        # Ensure all inputs are properly shaped
        batch_size = z_prev.size(0)
        
        # Handle different types of tensors and ensure proper dimensionality
        # For each tensor, we'll extract the feature dimension based on specific conditions
        
        if len(z_prev.shape) == 2:  # Already 2D [batch_size, features]
            z_prev_features = z_prev
        elif len(z_prev.shape) == 4:  # Convolutional features [batch_size, channels, height, width]
            z_prev_features = z_prev.reshape(batch_size, -1)
        else:
            raise ValueError(f"Unexpected shape for z_prev: {z_prev.shape}")
            
        if len(z_present.shape) == 2:  # Already 2D
            z_present_features = z_present
        elif len(z_present.shape) == 4:  # Convolutional features
            z_present_features = z_present.reshape(batch_size, -1)
        else:
            raise ValueError(f"Unexpected shape for z_present: {z_present.shape}")
            
        if len(z_serv.shape) == 2:  # Already 2D
            z_serv_features = z_serv
        elif len(z_serv.shape) == 4:  # Convolutional features
            z_serv_features = z_serv.reshape(batch_size, -1)
        else:
            raise ValueError(f"Unexpected shape for z_serv: {z_serv.shape}")
        
        # Normalize features
        z_prev_features = F.normalize(z_prev_features, p=2, dim=1)
        z_present_features = F.normalize(z_present_features, p=2, dim=1)
        z_serv_features = F.normalize(z_serv_features, p=2, dim=1)
        
        # Calculate cosine similarity directly using einsum for clarity and safety
        # This avoids issues with transpose operations
        sim_prev_present = torch.einsum('bi,bj->ij', z_prev_features, z_present_features) / self.temp
        sim_prev_serv = torch.einsum('bi,bj->ij', z_prev_features, z_serv_features) / self.temp
        
        # Compute contrastive loss
        logits = torch.cat([sim_prev_present, sim_prev_serv], dim=1)
        contrastive_labels = torch.arange(batch_size, device=z_prev.device)
        loss = F.cross_entropy(logits, contrastive_labels)
        
        # Apply dynamic divergence penalty
        if self.dynamic_beta:
            loss = loss * self.beta
        
        # Apply class-aware divergence penalty (if labels are provided)
        if self.class_aware_beta and labels is not None:
            # Make sure labels has the right shape
            if len(labels.shape) > 1:
                labels = labels.squeeze()
                
            class_counts = torch.bincount(labels, minlength=labels.max().item() + 1)
            class_weights = 1.0 / (class_counts[labels] + 1e-6)  # Inverse frequency as weights
            weighted_loss = loss * class_weights.mean()
            loss = weighted_loss
        
        return loss
