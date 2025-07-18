import torch
import torch.nn as nn
import torch.nn.functional as F 

def infoNCELoss(scores, labels_weights, temperature=0.1):
    """
    Contrastive loss over matching score. Adapted from https://arxiv.org/pdf/2004.11362.pdf Eq.2
    labels_weights: [B, N_scores] - flattened ground truth weights for each score position
                    (e.g., from the ground truth heatmap flattened and possibly oriented)
    scores: [B, N_scores] - flattened matching scores
    temperature: tau
    
    NOTE: This implementation assumes `labels_weights` correctly represents the w_k_i,j,r
          from the CCVPE paper (equation 2), where N_scores is the total number of
          possible matching positions/orientations per batch item.
    """
    
    # Scale scores to prevent overflow in exp, using detach to avoid gradients flowing back through max
    scores = scores / (torch.max(scores).detach() + 1e-8) 

    exp_scores = torch.exp(scores / temperature)
    
    # Denominator: sum over all possible samples for each item in the batch
    denominator = torch.sum(exp_scores, dim=1, keepdim=True)
    
    # Identify positive samples based on their weights
    bool_mask = labels_weights > 1e-6 
    
    # Calculate inner_element only for positive samples
    # log( exp(score_pos/temperature) / sum(exp(score/temperature)) )
    
    # Expand denominator to match the shape of masked_select for element-wise division
    masked_exp_scores = torch.masked_select(exp_scores, bool_mask)
    masked_denominator = torch.masked_select(denominator.expand_as(scores), bool_mask)
    
    # Handle potential division by zero if masked_denominator is zero (though unlikely with 1e-8 smooth)
    inner_element = torch.log(masked_exp_scores / (masked_denominator + 1e-8))
    
    # loss = - 1/sum(weights) * sum(inner_element*weights)
    masked_labels_weights = torch.masked_select(labels_weights, bool_mask)
    sum_positive_weights = torch.sum(masked_labels_weights)
    
    # Avoid division by zero if no positive samples are found (shouldn't happen in training)
    if sum_positive_weights == 0:
        return torch.tensor(0.0, device=scores.device)
        
    loss = -torch.sum(inner_element * masked_labels_weights) / sum_positive_weights
    
    return loss


def cross_entropy_loss(logits, labels):
    """
    Localization classification loss (Eq. 3 in CCVPE paper)
    logits: [B, L*L] - flattened localization prediction before softmax
    labels: [B, L*L] - flattened ground truth heatmap (Gaussian distribution), normalized to sum to 1
    """
    # log_softmax(logits) gives log probabilities
    log_probs = F.log_softmax(logits, dim=1) 
    # - sum(labels * log_probs) is the cross-entropy for each batch item
    # Then average over the batch
    loss = -torch.sum(labels * log_probs) / logits.size()[0] 
    return loss

def focal_loss(logits, labels, alpha=0.25, gamma=2.0):
    """
    Focal Loss for localization heatmap.
    Args:
        logits: [N, H*W], 模型的原始輸出 (flattened heatmap logits)
        labels: [N, H*W], 歸一化後的真實熱圖 (flattened heatmap), 範圍通常在 [0, 1]
        alpha: 平衡正負樣本的權重，alpha > 0.5 增加正樣本權重
        gamma: 聚焦參數，gamma > 0 減少易分類樣本的權重，使模型更關注難分類樣本
    Returns:
        損失值
    """
    # 將 logits 轉換為機率
    prob = torch.sigmoid(logits) 
    
    # 計算 BCE 損失，不進行歸約，以逐個元素處理
    # labels (gt_flattened) 是軟標籤，F.binary_cross_entropy_with_logits 接受軟標籤
    ce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    
    # 計算 p_t，這是原始論文中用於權重的機率
    # 對於正樣本 (labels=1)，p_t = prob
    # 對於負樣本 (labels=0)，p_t = 1 - prob
    # 對於軟標籤 (labels in [0,1])，則 p_t = labels * prob + (1 - labels) * (1 - prob)
    p_t = prob * labels + (1 - prob) * (1 - labels)
    
    # 計算 alpha_t，用於平衡正負樣本
    alpha_factor = labels * alpha + (1 - labels) * (1 - alpha)
    
    # 計算 modulating factor，用於減少易分類樣本的權重
    modulating_factor = (1 - p_t).pow(gamma)
    
    # Focal Loss = alpha_factor * modulating_factor * CE_loss
    loss = alpha_factor * modulating_factor * ce_loss
    
    # 對所有像素的損失求和，然後除以 batch_size 進行平均
    # (或者也可以考慮除以 `labels.sum()` 如果您希望對有效像素歸一化)
    return torch.sum(loss) / logits.size()[0]


def orientation_loss(predicted_ori_field, gt_orientation_vector, gt_heatmap_2d):
    """
    Orientation regression loss (Eq. 5 in CCVPE paper)
    predicted_ori_field: [B, H, W, 2] - predicted orientation vector field (cos, sin)
    gt_orientation_vector: [B, 2] - ground truth orientation vector (cos, sin) for each batch
    gt_heatmap_2d: [B, 1, H, W] - 2D Gaussian ground truth heatmap for localization
    """
    
    # Reshape gt_orientation_vector to [B, 1, 1, 2] to broadcast across H, W
    gt_ori_expanded = gt_orientation_vector.unsqueeze(1).unsqueeze(1) # B x 1 x 1 x 2 

    # (cos(o_gt) - Y1)^2 + (sin(o_gt) - Y2)^2
    squared_diff = torch.square(gt_ori_expanded - predicted_ori_field) # B x H x W x 2
    sum_squared_diff = torch.sum(squared_diff, dim=-1, keepdim=True) # B x H x W x 1

    # Multiply with D_gt (gt_heatmap_2d) and sum
    # gt_heatmap_2d is B x 1 x H x W, need to match B x H x W x 1
    gt_heatmap_broadcast = gt_heatmap_2d.permute(0, 2, 3, 1) # B x H x W x 1
    
    weighted_loss = sum_squared_diff * gt_heatmap_broadcast # B x H x W x 1
    
    # Sum over H and W, then average over batch
    loss = torch.sum(weighted_loss) / predicted_ori_field.size()[0] # Divide by batch size
    return loss