# Fashion-MNIST Classifier

A modular deep learning repository currently under active development, designed for classifying fashion items using the Fashion-MNIST dataset with PyTorch and Streamlit.

## Project Vision

Bridging the gap between exploratory data analysis and deployable deep learning applications through strict modularity, test-driven development, and a clean codebase.

## Development Architecture

The codebase is organized into modular directories to separate data processing, modeling, training, and deployment logic:

```
fashion-mnist-classifier/
├── app/
│   └── streamlit_app.py   # Interactive web interface for inference
├── data/                  # Dataset storage and management
├── models/                # Saved model weights and checkpoints
├── notebooks/
│   └── exploration.ipynb  # Exploratory data analysis and prototyping
├── src/
│   ├── __init__.py
│   ├── data.py            # Data loading and preprocessing pipelines
│   ├── model.py           # Neural network architecture definitions
│   ├── predict.py         # Inference execution logic
│   ├── train.py           # Training and evaluation loops
│   └── visualize.py       # Plotting and visualization utilities
├── tests/
│   ├── __init__.py
│   ├── test_data.py       # Unit tests for data module
│   ├── test_model.py      # Unit tests for model architecture
│   └── test_predict.py    # Unit tests for inference pipeline
├── .gitignore             # Git ignore rules
├── requirements.txt       # Project dependencies
└── README.md
```
