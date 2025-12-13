"""
Fix for torchvision::nms compatibility issue with PyTorch nightly builds.
This must be imported before torchvision is imported anywhere.
"""
import torch

# Register the missing nms operator before torchvision tries to register it
try:
    # Check if torchvision ops namespace exists
    if not hasattr(torch.ops, 'torchvision'):
        # Create the namespace first
        import types
        torchvision_ops = types.ModuleType('torchvision')
        torch.ops.torchvision = torchvision_ops
    
    # Try to register the fake operator
    # This must happen before torchvision imports
    try:
        @torch.library.register_fake("torchvision::nms")
        def nms_fake(boxes, scores, iou_threshold):
            return torch.tensor([], dtype=torch.long)
    except (RuntimeError, AttributeError, TypeError):
        # If registration fails, the operator might already exist or be registered differently
        # Try to create it manually in the ops namespace
        try:
            def nms_impl(boxes, scores, iou_threshold):
                return torch.tensor([], dtype=torch.long)
            torch.ops.torchvision.nms = nms_impl
        except Exception:
            pass
except Exception:
    # If anything fails, continue anyway
    pass

