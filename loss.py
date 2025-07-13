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


import torch
import os
import torch.nn.functional as F
MAX_FLOW = 400

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid_ = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid_[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid_.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def disp_pyramid_loss(disp_preds, disp_gt, gt_mask, pseudo_gt_disp, pseudo_mask, load_pseudo_gt=False):
    """ Loss function defined over sequence of flow predictions """
    pyramid_loss = []
    pseudo_pyramid_loss = []
    disp_loss = 0
    pseudo_disp_loss = 0
    # Loss weights
    if len(disp_preds) == 5:
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
    elif len(disp_preds) == 4:
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0]
    elif len(disp_preds) == 3:
        pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
    elif len(disp_preds) == 1:
        pyramid_weight = [1.0]  # highest loss only
    else:
        raise NotImplementedError

    for k in range(len(disp_preds)):
        pred_disp = disp_preds[k]
        weight = pyramid_weight[k]

        if pred_disp.size(-1) != disp_gt.size(-1):
            pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
            pred_disp = F.interpolate(pred_disp, size=(disp_gt.size(-2), disp_gt.size(-1)),
                                        mode='bilinear', align_corners=False) * (disp_gt.size(-1) / pred_disp.size(-1))
            pred_disp = pred_disp.squeeze(1)  # [B, H, W]

        curr_loss = F.smooth_l1_loss(pred_disp[gt_mask], disp_gt[gt_mask],
                                        reduction='mean')
        disp_loss += weight * curr_loss
        pyramid_loss.append(curr_loss)

        # Pseudo gt loss
        if load_pseudo_gt:
            pseudo_curr_loss = F.smooth_l1_loss(pred_disp[pseudo_mask], pseudo_gt_disp[pseudo_mask],
                                                reduction='mean')
            pseudo_disp_loss += weight * pseudo_curr_loss

            pseudo_pyramid_loss.append(pseudo_curr_loss)

    total_loss = disp_loss + pseudo_disp_loss
    metrics = {
        'disp_epe': pyramid_loss[-1],
    }

    return total_loss, metrics



def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()




def motion_consis_loss(flow_preds, rigid_flow_preds, mask, max_flow=300):
    """ Loss function defined over sequence of flow predictions """
    # flow_loss = 0.0
    motion_consis = 0.0
    loss_weight = 0.05

    mag = torch.sum(rigid_flow_preds**2, dim=1).sqrt()
    mask_ = (mask >= 0.5) & (mag < max_flow)

    motion_consis = (flow_preds[-1] - rigid_flow_preds).abs()
    motion_consis = loss_weight * (mask_[:, None] * motion_consis).mean()

    motion_epe = loss_weight * torch.sum((flow_preds[-1] - rigid_flow_preds)**2, dim=1).sqrt()
    motion_epe = motion_epe.view(-1)[mask_.view(-1)]

    metrics = {
        'motion_consis': motion_epe.mean().item(),
    }

    return motion_consis, metrics



def cross_domain_consis_loss(foggy_flow_preds, clean_flow_preds):
    """ Loss function defined over sequence of flow predictions """
    # flow_loss = 0.0

    loss_weight = 0.4
    loss = (foggy_flow_preds - clean_flow_preds).abs()
    loss = loss_weight * loss.mean()

    motion_epe = loss_weight * torch.sum((foggy_flow_preds - clean_flow_preds)**2, dim=1).sqrt()
    motion_epe = motion_epe.view(-1)

    metrics = {
        'foggy_domain_epe': motion_epe.mean().item(),
        'foggy_domain_1px': (motion_epe < 1).float().mean().item(),
        'foggy_domain_3px': (motion_epe < 3).float().mean().item(),
        'foggy_domain_5px': (motion_epe < 5).float().mean().item(),
    }

    return loss, metrics



def self_supervised_loss(flow_preds, flow_pseudo):
    """ Loss function defined over sequence of flow predictions """
    # flow_loss = 0.0

    loss_weight = 0.5
    loss = (flow_preds - flow_pseudo).abs()
    loss = loss_weight * loss.mean()

    motion_epe = loss_weight * torch.sum((flow_preds - flow_pseudo)**2, dim=1).sqrt()
    motion_epe = motion_epe.view(-1)

    metrics = {
        'pseudo_epe': motion_epe.mean().item(),
        'pseudo_1px': (motion_epe < 1).float().mean().item(),
        'pseudo_3px': (motion_epe < 3).float().mean().item(),
        'pseudo_5px': (motion_epe < 5).float().mean().item(),
    }

    return loss, metrics


# tensor_b: 指导作用的
def loss_KL_div(tensor_a, tensor_b, reduction='mean'):
    log_a = F.log_softmax(tensor_a)
    softmax_b = F.softmax(tensor_b,dim=-1)

    kl_mean = F.kl_div(log_a, softmax_b, reduction=reduction)

    return kl_mean
