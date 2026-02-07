## Sentiment Analysis Notebook

This project trains a sentiment classifier on the IMDB dataset using a BiLSTM
with GloVe word embeddings. The notebook includes data loading, preprocessing,
training with a train/validation split, and loss/accuracy plots.

## Files

- `sentiment_analysis.ipynb`: Main notebook.
- `aclImdb/`: IMDB dataset folder (after extracting).
- `glove.6B.100d.txt`: GloVe embeddings (100d).

## Requirements

- Python 3.11
- PyTorch
- NumPy
- Matplotlib

On Apple Silicon, the notebook uses MPS if available.

## Data Setup

1) IMDB dataset:

- Download `aclImdb_v1.tar.gz`
- Extract so you have:
  - `aclImdb/train/pos`, `aclImdb/train/neg`
  - `aclImdb/test/pos`, `aclImdb/test/neg`

2) GloVe:

- Download GloVe 6B
- Place `glove.6B.100d.txt` in the project root, or update the path in the
  notebook.

## Run

Open `sentiment_analysis.ipynb` and run cells top-to-bottom.

Key parameters you can tune:

- `EMB_DIM` (must match the GloVe file)
- `HIDDEN_DIM`
- `batch_size` in the DataLoaders
- `max_len` in `IMDBDataset`
- training epochs

## Outputs

- Train/validation loss and accuracy printed each epoch
- Loss and accuracy plots at the end

## Notes

- The dataset is split 90/10 into train/validation.
- The model uses mean+max pooling over BiLSTM outputs.
- Test set loading is included for later evaluation.
