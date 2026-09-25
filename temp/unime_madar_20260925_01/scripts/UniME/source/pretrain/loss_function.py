import torch
from torch import nn


class ReconstructionLoss(nn.Module):
    """
    Whole-volume MSE plus the raw learnable-mask L2 penalty (paper Equation 5).

    <NOTE>:
        - Add the released code's epsilon to each spatial L2 norm, then average
          over batch and modality; the paper does not specify these reductions.
        - Do not subtract spatial means: constant nonzero masks are penalized too.
    """
    def __init__(self, regulization_rate: float = 0.005, eps: float = 1e-6):
        """
        Args:
            regulization_rate (float, optional): Coefficient of the mask L2 penalty.
            eps (float, optional): Released code's additive offset to spatial norms.
                It is outside the norm and therefore does not change its gradient.
        """
        super().__init__()
        self.regularization_rate = regulization_rate
        self.mse = nn.MSELoss(reduction='mean')
        self.eps = eps

    def forward(
        self, recon: torch.Tensor, learnable_prior: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            recon (torch.Tensor): Reconstruction output with shape (B, num_modalities, D, H, W)
            learnable_prior (torch.Tensor): Learnable prior output with shape (B, num_modalities, D, H, W)
            target (torch.Tensor): Target output with shape (B, num_modalities, D, H, W)

        <NOTE>:
            - learnable_prior is the learnable prior output of the model.
            - recon is assumed <WITHOUT> SoftMax or LogSoftMax.

        Returns:
            torch.Tensor: scalar loss
        """
        match self.regularization_rate:
            case 0.0:
                return self.mse(recon, target)
            case _:
                reg = torch.linalg.vector_norm(learnable_prior, ord=2, dim=(2, 3, 4))
                reg = (reg + self.eps).mean()
                return self.mse(recon, target) + self.regularization_rate * reg
