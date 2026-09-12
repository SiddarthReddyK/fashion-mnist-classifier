import torch
import torch.nn as nn
import torch.optim as optim

from src.data import get_dataloaders
from src.model import SimpleCNN


def train_model(epochs=5, batch_size=64, lr=0.001, save_path="models/fashion_mnist_cnn.pt"):
    train_loader, test_loader = get_dataloaders(batch_size = batch_size)

    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)

    losses = []       
    accuracies = []
     
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = running_loss / len(train_loader)
        accuracy = 100* correct / total
        losses.append(avg_loss)         # NEW
        accuracies.append(accuracy)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, test_loader, losses, accuracies

if __name__ == "__main__":
    from src.visualize import plot_training_curve, plot_confusion_matrix, plot_sample_predictions

    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    model, test_loader, losses, accuracies = train_model()

    plot_training_curve(losses, accuracies)
    plot_confusion_matrix(model, test_loader, class_names)
    plot_sample_predictions(model, test_loader, class_names)