import torch 

def IoU(b1,b2,size):
    x1_1 = b1[0]
    y1_1 = b1[1]
    x2_1 = b1[0] + size[0]
    y2_1 = b1[1] + size[1]

    x1_2 = b2[0]
    y1_2 = b2[1]
    x2_2 = b2[0] + size[0]
    y2_2 = b2[1] + size[1]

    xA = max(x1_1,x1_2)
    xB = min(x2_1,x2_2)
    yA = max(y1_1,y1_2)
    yB = min(y2_1,y2_2)

    inter = max(0,xB-xA) * max(0,yB-yA)

    return inter > 0.0

def box_iou(boxes1, boxes2):

    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas > 0.0

