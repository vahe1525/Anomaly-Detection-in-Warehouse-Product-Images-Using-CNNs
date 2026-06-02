"""
Wood Anomaly Detection — Prediction / Inference Script
=======================================================
Loads a model that was trained and saved by main.py, then classifies
brand-new wood images as Good or Anomaly. No training happens here.

First train and save a model by running main.py (it saves cnn_model.pth
or resnet_model.pth). Then use this script.

Usage:
    python predict.py path/to/image.png
    python predict.py path/to/folder/        (classifies every image in the folder)

Switch which trained model to use with MODEL_TYPE below — it must match a
.pth file that main.py already produced.
"""

# ── Choose which trained model to load ────────────────────────────────────────
MODEL_TYPE = 'resnet'   # 'cnn'  or  'resnet'

import os
import sys
import torch
from PIL import Image
from torchvision import transforms

from model import WoodAnomalyDetector           # custom CNN
from resnet_model import WoodResNetDetector     # ResNet18 transfer learning


eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(model_type, device):
    """
    Build the right empty model, then load the saved weights into it.

    load_state_dict fills the freshly built network with the weights main.py saved.
    model.eval() switches off Dropout so the full network is used for prediction.
    """
    if model_type == 'resnet':
        model = WoodResNetDetector(pretrained=False)   # no need to download ImageNet
        weights_path = 'resnet_model.pth'              # weights — our trained ones come next
    else:
        model = WoodAnomalyDetector()
        weights_path = 'cnn_model.pth'

    if not os.path.exists(weights_path):
        sys.exit(f"Error: '{weights_path}' not found. Train first with: python main.py "
                 f"(set MODEL_TYPE = '{model_type}')")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Loaded {model_type.upper()} model from {weights_path}")
    return model


def predict_image(model, image_path, device):
    """
    Classify a single image.
    Returns (label_name, probability) where probability is the model's
    confidence that the image is an Anomaly (0.0 -> Good, 1.0 -> Anomaly).
    """
    image  = Image.open(image_path).convert('RGB')
    # unsqueeze(0) adds the batch dimension: [3,224,224] -> [1,3,224,224]
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(tensor).item()   # single float in [0, 1]

    label = 'Anomaly' if prob > 0.5 else 'Good'
    return label, prob


def collect_image_paths(target):
    """Return a list of image paths from either a single file or a folder."""
    valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.avif')
    if os.path.isdir(target):
        paths = [os.path.join(target, f) for f in sorted(os.listdir(target))
                 if f.lower().endswith(valid_ext)]
        if not paths:
            sys.exit(f"No images found in folder: {target}")
        return paths
    if os.path.isfile(target):
        return [target]
    sys.exit(f"Path not found: {target}")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python predict.py <image_or_folder_path>")

    target = sys.argv[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model       = load_model(MODEL_TYPE, device)
    image_paths = collect_image_paths(target)

    print(f"\nClassifying {len(image_paths)} image(s):\n")
    for path in image_paths:
        label, prob = predict_image(model, path, device)
        # Confidence: how sure the model is about the label it chose
        confidence = prob if label == 'Anomaly' else 1 - prob
        name = os.path.basename(path)
        print(f"  {name:<30} -> {label:<8} (anomaly prob: {prob:.3f}, confidence: {confidence:.1%})")


if __name__ == '__main__':
    main()
