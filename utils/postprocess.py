import torch


def postprocess(outputs, target_size, queries=True):
    if queries:
        masks_classes = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
        masks_logits = outputs.masks_queries_logits
    else:
        masks_logits = outputs.logits
    masks_logits = torch.nn.functional.interpolate(masks_logits, size=target_size, mode="bilinear", align_corners=False)

    masks_probs = masks_logits.sigmoid()
    if queries:
        masks_probs = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
    return masks_probs
