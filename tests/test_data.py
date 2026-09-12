import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data import get_dataloaders


def test_batch_shape():
    train_loader, _ = get_dataloaders(batch_size=32)
    images, labels = next(iter(train_loader))
    assert images.shape == (32, 1, 28, 28)


def test_labels_in_valid_range():
    train_loader, _ = get_dataloaders(batch_size=32)
    images, labels = next(iter(train_loader))
    assert labels.min() >= 0
    assert labels.max() <= 9


def test_train_and_test_sets_different_sizes():
    train_loader, test_loader = get_dataloaders()
    assert len(train_loader.dataset) == 60000
    assert len(test_loader.dataset) == 10000