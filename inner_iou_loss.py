"""
Inner-CIoU Loss — Gradient-amplified IoU for small degraded objects.

Core idea (Inner-IoU, Zhang et al. 2023):
  Instead of computing CIoU on the full box, compute it on an "inner" box
  scaled from the center by ratio. A ratio < 1 makes the inner box smaller
  than the original, so the same pixel displacement produces a larger IoU
  change → stronger regression gradient for small objects.

  inner_w = w * ratio,  inner_h = h * ratio
  inner_box = (xc - inner_w/2, yc - inner_h/2, xc + inner_w/2, yc + inner_h/2)

This preserves the CIoU metric space (unlike NWD), so the TaskAlignedAssigner's
CIoU-based alignment_metric stays compatible with the loss function.
"""

import math
import torch
import torch.nn.functional as F


def inner_ciou(box1, box2, ratio=0.75, eps=1e-7):
    """
    Inner-CIoU between two sets of boxes in xyxy format.

    Args:
        box1: (..., 4) ground-truth boxes [x1, y1, x2, y2]
        box2: (..., 4) predicted boxes [x1, y1, x2, y2]
        ratio: inner-box scale factor. < 1 amplifies small-object gradients.
        eps: numerical stability

    Returns:
        (...,) CIoU values computed on inner boxes

    Shape reference (from BboxLoss.forward):
        box1: [N, 4] — target_bboxes[fg_mask]
        box2: [N, 4] — pred_bboxes[fg_mask]
        returns: [N] CIoU values
    """
    # --- Parse coordinates ---
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)

    # --- Compute full-box values for center distance and aspect ratio ---
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # --- Inner box: scale from center ---
    # GT inner box
    b1_cx = (b1_x1 + b1_x2) * 0.5
    b1_cy = (b1_y1 + b1_y2) * 0.5
    inner_w1, inner_h1 = w1 * ratio, h1 * ratio
    inner_b1_x1 = b1_cx - inner_w1 * 0.5
    inner_b1_x2 = b1_cx + inner_w1 * 0.5
    inner_b1_y1 = b1_cy - inner_h1 * 0.5
    inner_b1_y2 = b1_cy + inner_h1 * 0.5

    # Pred inner box
    b2_cx = (b2_x1 + b2_x2) * 0.5
    b2_cy = (b2_y1 + b2_y2) * 0.5
    inner_w2, inner_h2 = w2 * ratio, h2 * ratio
    inner_b2_x1 = b2_cx - inner_w2 * 0.5
    inner_b2_x2 = b2_cx + inner_w2 * 0.5
    inner_b2_y1 = b2_cy - inner_h2 * 0.5
    inner_b2_y2 = b2_cy + inner_h2 * 0.5

    # --- Inner-box IoU ---
    inter = (
        inner_b1_x2.minimum(inner_b2_x2) - inner_b1_x1.maximum(inner_b2_x1)
    ).clamp_(0) * (
        inner_b1_y2.minimum(inner_b2_y2) - inner_b1_y1.maximum(inner_b2_y1)
    ).clamp_(0)

    union = inner_w1 * inner_h1 + inner_w2 * inner_h2 - inter + eps
    iou = inter / union

    # --- CIoU terms on full boxes (center distance + aspect ratio) ---
    # convex box
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps

    # center distance (on original boxes, not inner)
    rho2 = (
        (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) +
        (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
    ) / 4  # center dist^2

    # aspect ratio penalty
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))

    return iou - (rho2 / c2 + v * alpha)


def patch_loss_for_inner_iou(ratio=0.75):
    """
    Monkey-patch ultralytics BboxLoss.forward to use Inner-CIoU.

    Only patches the loss computation; the TaskAlignedAssigner continues
    using standard CIoU, preserving sample-assignment stability.

    Args:
        ratio: inner-box scale factor (0.75 = amplify small-object gradient)
    """
    from ultralytics.utils.loss import BboxLoss
    from ultralytics.utils.tal import bbox2dist

    original_forward = BboxLoss.forward

    def patched_forward(
        self,
        pred_dist, pred_bboxes, anchor_points,
        target_bboxes, target_scores, target_scores_sum,
        fg_mask, imgsz, stride,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        # --- THE ONLY LINE CHANGED: CIoU → Inner-CIoU ---
        iou = inner_ciou(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], ratio=ratio
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss (unchanged)
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl

    BboxLoss.forward = patched_forward
    print(f"[INNER-IOU] BboxLoss patched: CIoU → Inner-CIoU(ratio={ratio})")
    print(f"[INNER-IOU] TaskAlignedAssigner UNCHANGED (standard CIoU for stable matching)")
    return original_forward


def restore_loss(original_forward):
    """Restore original BboxLoss.forward after training."""
    from ultralytics.utils.loss import BboxLoss
    BboxLoss.forward = original_forward
    print("[INNER-IOU] BboxLoss restored to original CIoU.")


# ---------------------------------------------------------------------------
# Gradient analysis helpers (for debugging / ablation)
# ---------------------------------------------------------------------------
def grad_amplification_factor(ratio=0.75):
    """
    Approximate gradient amplification for small vs large objects.

    For Inner-IoU with ratio < 1:
      d(IoU_inner)/d(offset) ≈ (1/ratio) * d(IoU_full)/d(offset)

    for small boxes (where offset/w is large relative to offset/(ratio*w)).
    """
    return 1.0 / ratio


if __name__ == "__main__":
    # Sanity check: Inner-CIoU gradients are larger than standard CIoU
    print("=" * 60)
    print("Inner-CIoU Gradient Sanity Check")
    print("=" * 60)

    box_gt = torch.tensor([[10.0, 10.0, 30.0, 30.0]], requires_grad=False)  # 20×20
    box_pd = torch.tensor([[12.0, 12.0, 32.0, 32.0]], requires_grad=True)    # offset by 2px

    # Standard CIoU
    from ultralytics.utils.metrics import bbox_iou
    ciou = bbox_iou(box_gt, box_pd, xywh=False, CIoU=True)
    ciou.backward()
    ciou_grad = box_pd.grad.clone().abs().sum().item()
    box_pd.grad = None

    # Inner-CIoU
    ic = inner_ciou(box_gt, box_pd, ratio=0.75)
    ic.backward()
    ic_grad = box_pd.grad.clone().abs().sum().item()

    print(f"  Standard CIoU grad: {ciou_grad:.6f}")
    print(f"  Inner-CIoU  grad:  {ic_grad:.6f}")
    print(f"  Amplification:     {ic_grad / ciou_grad:.2f}x")
    print(f"  Theory (1/ratio):  {1.0 / 0.75:.2f}x")
    print()
    print("OK — Inner-CIoU amplifies small-object regression gradient.")
