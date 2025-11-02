# Named Entity Recognition with Bi-LSTM

Trained a bi-directional LSTM model using Keras with TensorFlow at its backend to identify entities in literature.

## Overview

This project implements a Named Entity Recognition (NER) system using deep learning techniques. The model uses a Bi-directional LSTM architecture with advanced regularization techniques to achieve high accuracy in identifying named entities in text data. The project also demonstrates transfer learning by fine-tuning the model on biomedical datasets.

## Key Features

• **Bi-directional LSTM Architecture**: Implemented using Keras with TensorFlow backend for robust entity recognition in literature and biomedical texts.

• **Advanced Regularization**: Incorporated a one-dimensional spatial dropout layer to reduce overfitting, improving model generalization.

• **Optimizer**: Used the Adam optimizer to achieve an accuracy of 98%.

• **Transfer Learning**: Fine-tuned the model on the BC5CDR dataset to identify biomedical named entities (Chemical and Disease) using transfer learning techniques.

## Model Architecture

- **Embedding Layer**: Word embeddings with configurable dimensions
- **Spatial Dropout 1D**: Regularization layer to prevent overfitting
- **Bidirectional LSTM**: Captures context from both directions
- **Time Distributed Dense Layer**: Output layer for sequence labeling

## Results

- **Accuracy**: 98% on general literature NER task
- **Entity F1-Score**: 54.81% on BC5CDR biomedical dataset (entities only, excluding 'O')
- **Weighted F1-Score**: 70.74% on BC5CDR biomedical dataset

## Requirements

The project requires the following Python packages:

```
tensorflow>=2.0.0
keras
pandas
numpy
matplotlib
scikit-learn
nltk
livelossplot
```

## Usage

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Named-Entity-Recognition.git
cd Named-Entity-Recognition
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook Named_Entity_Recognition.ipynb
```

## Project Structure

```
Named-Entity-Recognition/
├── Named_Entity_Recognition.ipynb  # Main notebook with implementation
├── bc5cdr/                          # BC5CDR dataset directory
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
└── .gitignore                        # Git ignore file
```

## Notebook Contents

The notebook includes:
- Data preprocessing
- Bi-LSTM model architecture
- Training with callbacks (early stopping, loss plotting)
- Model evaluation with custom metrics (F1, precision, recall)
- Transfer learning: Fine-tuning on BC5CDR biomedical dataset

## Dataset

The project uses a standard NER dataset for initial training and the BC5CDR dataset for biomedical entity recognition fine-tuning.

## License

This project is open source and available under the MIT License.