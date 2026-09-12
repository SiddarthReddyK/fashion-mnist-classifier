import torch
from PIL import Image
from torchvision import transforms

from src.model import SimpleCNN

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_model(model_path="models/fashion_mnist_cnn.pt"):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict_image(model, image: Image.Image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(image).unsqueeze(0)  # add batch dimension

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    return predicted_class, confidence.item()


if __name__ == "__main__":
    model = load_model()
    from src.data import get_dataloaders
    _, test_loader = get_dataloaders(batch_size=1)
    images, labels = next(iter(test_loader))

    from torchvision.transforms.functional import to_pil_image
    img = to_pil_image(images[0])

    predicted_class, confidence = predict_image(model, img)
    print(f"Predicted: {predicted_class} ({confidence*100:.1f}% confidence)")
    print(f"Actual: {CLASS_NAMES[labels[0].item()]}")

