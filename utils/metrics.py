def direction_aware_iou(box1, box2, xywh=True, alpha_h=1.0, alpha_v=1.5, GIoU=False, DIoU=False, CIoU=True, eps=1e-7):
    """
    Calculate Direction-Aware IoU for road crack detection.

    Args:
        box1, box2: Boxes in [x, y, w, h] or [x1, y1, x2, y2] format
        xywh: If True, input boxes are in [x, y, w, h] format
        alpha_h: Weight for horizontal direction (for transverse cracks)
        alpha_v: Weight for vertical direction (for longitudinal cracks)
        GIoU, DIoU, CIoU: Additional IoU variants
        eps: Small value to prevent division by zero
    """
    with open("direction_iou_log.txt", "a") as f:
        f.write(f"direction_aware_iou called at {time.asctime()}\n")
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
            b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # Calculate aspect ratio for both boxes
    ar1 = w1 / (h1 + eps)  # aspect ratio of box1: width/height
    ar2 = w2 / (h2 + eps)  # aspect ratio of box2: width/height

    # Direction weight factors
    # Higher weight for boxes with high aspect ratio (horizontal/transverse cracks)
    horizontal_factor = torch.sigmoid(torch.max(ar1, ar2) - 1.0) * (alpha_h - 1.0) + 1.0
    # Higher weight for boxes with low aspect ratio (vertical/longitudinal cracks)
    vertical_factor = torch.sigmoid(1.0 - torch.min(ar1, ar2)) * (alpha_v - 1.0) + 1.0

    # Combine direction factors
    # direction_factor = torch.max(horizontal_factor, vertical_factor)
    alligator_factor = torch.min(horizontal_factor, vertical_factor) * 1.5  # 当两个方向都显著时
    direction_factor = (horizontal_factor + vertical_factor) / 2

    # Standard IoU
    iou = inter / union

    # Apply direction factor
    iou = iou * direction_factor

    if CIoU or DIoU or GIoU:
        # Same implementation as in the original bbox_iou function
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU
            c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
            rho2 = (
                           (b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)
                   ) / 4  # center dist squared
            if CIoU:  # Complete IoU
                v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU

    return iou  # Direction-aware IoU