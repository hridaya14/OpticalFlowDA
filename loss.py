import torch
import torch.nn.Functional as F


def compute_flow_loss(pred_flow, gt_flow, valid_mask=None):
    """
    Computes MSE loss and a dictionary of important metrics for optical flow evaluation,
    structured similarly to standard optical flow training scripts.

    Args:
        pred_flow (torch.Tensor): The predicted flow map (B, 2, H, W).
        gt_flow (torch.Tensor): The ground truth flow map (B, 2, H, W).
        valid_mask (torch.Tensor, optional): A boolean mask where True indicates a
                                             valid pixel for evaluation (B, H, W).

    Returns:
        tuple: A tuple containing:
               - mse_loss (torch.Tensor): The raw MSE loss tensor for backpropagation.
               - metrics (dict): A dictionary of metrics for logging (epe, 1px, 3px, 5px).
    """
    # Ensure tensors are on the same device and have the same shape
    assert pred_flow.shape == gt_flow.shape, "Input tensors must have the same shape"

    # --- 1. Calculate Mean Squared Error (MSE) Loss ---
    # This will be the loss value returned for backpropagation.
    mse_loss = F.mse_loss(pred_flow, gt_flow)

    # --- 2. Calculate Metrics (EPE and N-pixel accuracy) ---
    # Detach tensors from the computation graph for metric calculation to save memory.
    with torch.no_grad():
        # Calculate the per-pixel Euclidean distance (EPE map)
        epe_map = torch.sqrt(torch.sum((pred_flow - gt_flow)**2, dim=1))

        # If a valid mask is not provided, create one that considers all pixels valid
        if valid_mask is None:
            valid_mask = torch.ones_like(epe_map, dtype=torch.bool)

        # Filter the EPE map to only include valid pixels
        valid_epe = epe_map[valid_mask]

        # Calculate the average EPE over all valid pixels
        avg_epe = valid_epe.mean()

        # Calculate N-pixel accuracy metrics: the percentage of valid pixels
        # with an EPE of less than 1, 3, and 5 pixels, respectively.
        px1 = (valid_epe < 1).float().mean()
        px3 = (valid_epe < 3).float().mean()
        px5 = (valid_epe < 5).float().mean()

        metrics = {
            'epe': avg_epe.item(),
            '1px': px1.item(),
            '3px': px3.item(),
            '5px': px5.item(),
        }

    return mse_loss, metrics
