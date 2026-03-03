import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

MODEL_PATH = "C:\\Users\\dubba\\OneDrive\\Desktop\\Hackathon\\models\\best_model.pth"
IMG_PATH = "C:\\Users\\dubba\\OneDrive\\Desktop\\Hackathon\\testimages\\0c917c372572.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

classes = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate"]

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model(model_path):
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
def generate_gradcam(model, img_tensor, target_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, input, output):
        activations.append(output)
    for name, module in model.named_modules():
        if name == "features":
            module.register_forward_hook(forward_hook)
            module.register_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    loss = output[0, target_class]
    loss.backward()

    gradient = gradients[0][0].cpu().data.numpy()
    activation = activations[0][0].cpu().data.numpy()

    weights = np.mean(gradient, axis=(1, 2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * activation[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

def predict_image(image_path, model):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)

    pred_class = pred.item()
    print(f"✅ Predicted DR stage: {classes[pred_class]}")
    return img, img_tensor, pred_class

def visualize_gradcam(img, cam, pred_class):
    img_np = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("Original Retina Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(superimposed)
    plt.title(f"Grad-CAM → {classes[pred_class]}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_path = IMG_PATH

    model = load_model(MODEL_PATH)
    img, img_tensor, pred_class = predict_image(image_path, model)
    cam = generate_gradcam(model, img_tensor, pred_class)
    visualize_gradcam(img, cam, pred_class)
