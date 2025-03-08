import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temp=0.1, temperature=None, margin=1.0, dynamic_beta=False, 
                 beta_min=0.1, beta_max=1.0, beta_decay=0.99, class_aware_beta=False):
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
        """Update the dynamic beta based on the training progress."""
        if self.dynamic_beta:
            self.beta = self.beta_min + (self.beta_max - self.beta_min) * (self.beta_decay ** epoch)

    def forward(self, z_prev, z_present, z_serv, labels=None, epoch=None):
        """
        Compute the contrastive loss with dynamic and class-aware divergence penalties.

        Args:
            z_prev (torch.Tensor): Global features from the previous round.
            z_present (torch.Tensor): Local features from the current round.
            z_serv (torch.Tensor): Global features from the server.
            labels (torch.Tensor, optional): Class labels for class-aware divergence penalty.
            epoch (int, optional): Current epoch for dynamic beta update.

        Returns:
            torch.Tensor: Computed contrastive loss.
        """
        if epoch is not None:
            self.update_beta(epoch)  # Update beta based on the current epoch

        # Debugging: Print tensor shapes
        print(f"z_prev shape: {z_prev.shape}")
        print(f"z_present shape: {z_present.shape}")
        print(f"z_serv shape: {z_serv.shape}")

        # Ensure all tensors have shape [batch_size, feature_dim]
        batch_size, feature_dim = z_prev.shape[0], z_prev.shape[1]

        z_prev = z_prev.view(batch_size, feature_dim)
        z_present = z_present.view(batch_size, feature_dim)
        z_serv = z_serv.view(batch_size, feature_dim)

        # Normalize feature vectors
        z_prev = F.normalize(z_prev, p=2, dim=1)
        z_present = F.normalize(z_present, p=2, dim=1)
        z_serv = F.normalize(z_serv, p=2, dim=1)

        # Compute pairwise similarity matrices using .mT (correct transposition)
        sim_prev_present = torch.matmul(z_prev, z_present.mT) / self.temp
        sim_prev_serv = torch.matmul(z_prev, z_serv.mT) / self.temp

        # Debugging: Print similarity matrix shapes
        print(f"sim_prev_present shape: {sim_prev_present.shape}")
        print(f"sim_prev_serv shape: {sim_prev_serv.shape}")

        # Concatenate similarity matrices
        logits = torch.cat([sim_prev_present, sim_prev_serv], dim=1)

        # Create labels for contrastive loss
        labels = torch.arange(batch_size, device=z_prev.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        # Apply dynamic divergence penalty
        if self.dynamic_beta:
            loss = loss * self.beta

        # Apply class-aware divergence penalty (if labels are provided)
        if self.class_aware_beta and labels is not None:
            class_counts = torch.bincount(labels, minlength=batch_size)
            class_weights = 1.0 / (class_counts + 1e-6)  # Inverse frequency as weights
            loss = loss * class_weights[labels]

        return loss.mean()
