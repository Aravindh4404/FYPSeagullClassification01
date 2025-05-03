class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.tokens = None
        self.tokens_grad = None
        
        # Register hook in forward_features method or earlier in the pipeline
        def hook_fn(module, input, output):
            output.register_hook(self._save_grad)
            self.tokens = output
        
        # Find appropriate layer to hook (could be the output of forward_features or transformer blocks)
        self.model.vit.blocks[-1].register_forward_hook(hook_fn)
    
    def _save_grad(self, grad):
        self.tokens_grad = grad
    
    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        logits = self.model(input_tensor)
        
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()
        
        # Get patch tokens (skip CLS token)
        patch_tokens = self.tokens[0, 1:, :]
        patch_tokens_grad = self.tokens_grad[0, 1:, :]
        
        # Compute Grad-CAM
        weights = patch_tokens_grad.mean(dim=1, keepdim=True)
        cam = torch.matmul(patch_tokens, weights).reshape(14, 14)
        cam = F.relu(cam)
        
        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to image size
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        
        return cam, target_class, torch.softmax(logits, dim=1)[0, target_class].item()