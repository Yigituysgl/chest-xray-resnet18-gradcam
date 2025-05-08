import torch
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

def generate_gradcam(model, image_tensor, target_layer):
    model.eval()
    
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    handle_fw = target_layer.register_forward_hook(forward_hook)
    handle_bw = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor.unsqueeze(0))
    class_idx = output.argmax().item()
    model.zero_grad()
    output[0, class_idx].backward()

    grads = gradients[0]
    acts = activations[0]
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]
    
    heatmap = acts.mean(dim=1).squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    handle_fw.remove()
    handle_bw.remove()

    return heatmap
