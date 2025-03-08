import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1, margin=1.0, dynamic_beta=False, beta_min=0.1, beta_max=1.0, beta_decay=0.99, class_aware_beta=False):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.margin = margin
        self.dynamic_beta = dynamic_beta
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_decay = beta_decay
        self.class_aware_beta = class_aware_beta
        self.beta = beta_max  # Initialize beta to max value

    def update_beta(self, epoch):
        """Update dynamic beta based on training progress."""
        if self.dynamic_beta:
            self.beta = self.beta_min + (self.beta_max - self.beta_min) * (self.beta_decay ** epoch)

    def forward(self, z_prev, z_present, z_serv, labels=None, epoch=None):
        if epoch is not None:
            self.update_beta(epoch)  # Update beta based on epoch
        
        # 🛠 Fix: Ensure tensors are correctly reshaped for matmul
        batch_size = z_prev.shape[0]

        # Reshape to [batch_size, feature_dim] if necessary
        z_prev = z_prev.view(batch_size, -1)
        z_present = z_present.view(batch_size, -1)
        z_serv = z_serv.view(batch_size, -1)
        
        # Normalize features
        z_prev = F.normalize(z_prev, p=2, dim=1)
        z_present = F.normalize(z_present, p=2, dim=1)
        z_serv = F.normalize(z_serv, p=2, dim=1)
        
        # 🛠 Fix: Use .permute(1, 0) instead of .T to avoid PyTorch warnings
        sim_prev_present = torch.matmul(z_prev, z_present.permute(1, 0)) / self.temp
        sim_prev_serv = torch.matmul(z_prev, z_serv.permute(1, 0)) / self.temp
        
        # Compute contrastive loss using cross-entropy
        logits = torch.cat([sim_prev_present, sim_prev_serv], dim=1)
        contrastive_labels = torch.arange(batch_size, device=z_prev.device)
        loss = F.cross_entropy(logits, contrastive_labels)

        # Apply dynamic divergence penalty
        if self.dynamic_beta:
            loss = loss * self.beta
        
        # Apply class-aware divergence penalty
        if self.class_aware_beta and labels is not None:
            class_counts = torch.bincount(labels, minlength=labels.max().item() + 1)
            class_weights = 1.0 / (class_counts[labels] + 1e-6)  # Inverse frequency
            loss = (loss * class_weights).mean()
        
        return loss
