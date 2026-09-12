import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_training_curve(losses, accuracies, save_path="outputs/training_curve.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses, marker='o')
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(accuracies, marker='o', color='green')
    ax2.set_title("Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curve saved to {save_path}")

def plot_confusion_matrix(model, test_loader, class_names, save_path="outputs/confusion_matrix.png"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

    cm = torch.zeros(10, 10, dtype=torch.int32)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def plot_sample_predictions(model, test_loader, class_names, save_path="outputs/sample_predictions.png", num_samples=8):
    model.eval()
    images, labels = next(iter(test_loader))

    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i >= num_samples:
            break
        img = images[i].squeeze()
        true_label = class_names[labels[i]]
        pred_label = class_names[predicted[i]]
        color = 'green' if labels[i] == predicted[i] else 'red'

        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Sample predictions saved to {save_path}")

