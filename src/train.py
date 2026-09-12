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
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model, test_loader

if __name__ == "__main__":
    train_model()

