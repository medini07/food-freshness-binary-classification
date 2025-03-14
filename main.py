import torch
from data_processing import get_dataloaders
from model import get_model
from utils import load_model # type: ignore
from config import DATA_DIR, DEVICE
from torchvision import transforms
from PIL import Image



def predict(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

if __name__ == "__main__":
    _, test_loader, num_classes = get_dataloaders(DATA_DIR)
    model = get_model(num_classes).to(DEVICE)
    model = load_model(model, "fruit_veg_classifier.pth", DEVICE)

    # Example prediction
    test_transforms = test_loader.dataset.transform
    class_names = test_loader.dataset.classes
    image_path = "./dataset/test/sample_image.jpg"

    prediction = predict(model, image_path, test_transforms)
    print(f"Predicted class: {class_names[prediction]}")
