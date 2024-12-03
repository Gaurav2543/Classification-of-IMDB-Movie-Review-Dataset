# IMDB Movie Review Sentiment Analysis using RNNs

## Project Overview

This project implements a Recurrent Neural Network (RNN) model to predict sentiment (positive/negative) from IMDB movie reviews. The solution explores various neural network architectures and text preprocessing techniques to optimize sentiment classification performance.

## Key Features

- Text preprocessing with NLTK including stemming/lemmatization
- Multiple RNN architectures including vanilla RNN and LSTM
- Different output aggregation strategies (last hidden state vs mean pooling)
- Hyperparameter tuning experiments
- Comprehensive model evaluation and visualization

## Model Architecture Details

### Base RNN Model

- Word embedding layer
- RNN layer for sequence processing
- Linear output layer for binary classification
- Cross entropy loss function
- Adam optimizer

### Key Improvements

#### Text Preprocessing

- Implemented stemming and lemmatization using NLTK
- Created word-to-index mapping for vocabulary
- Limited reviews to 100-500 words for consistent processing
- Achieved 50.01% accuracy with stemming/lemmatization vs 50.96% without

#### Architecture Variants

1. Output Pooling

   - Last hidden state: 50.99% accuracy
   - Mean pooling: 84% accuracy
   - Mean pooling showed significant improvement by utilizing information from all timesteps
2. LSTM Enhancement

   - Replaced RNN layer with LSTM layer
   - Added gates for better long-term dependency modeling
   - Similar performance to RNN suggesting dataset may not require complex long-term modeling

#### Hyperparameter Tuning

- Experimented with:
  - Embedding dimensions
  - Hidden state size
  - Batch size
  - Learning rate
- Best configuration:
  - Embedding dim: 128
  - Hidden size: 256
  - Batch size: 32
  - Learning rate: 0.001

## Key Results

- Mean pooling significantly outperformed last-hidden-state approach
- Stemming/lemmatization slightly decreased accuracy
- LSTM performed similarly to RNN for this dataset
- Achieved best test accuracy of 84.12% with tuned hyperparameters

## Implementation Details

The solution uses PyTorch for model implementation with the following key components:

- Custom Dataset class for IMDB data processing
- DataLoader for batch processing
- Model classes for different architectures
- Training and evaluation loops
- Visualization utilities for performance metrics

## Conclusion

The project demonstrates the effectiveness of RNNs for sentiment analysis while highlighting important architectural considerations. The superior performance of mean pooling suggests that utilizing information from all timesteps is crucial for this task. The similar performance between RNN and LSTM indicates that sophisticated handling of long-term dependencies may not be essential for this particular dataset.
