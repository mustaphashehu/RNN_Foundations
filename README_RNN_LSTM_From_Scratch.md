# RNN and LSTM Forward and Backward Propagation in NumPy

## Aim

To understand how recurrent neural networks process sequences and propagate gradients across time by implementing **vanilla RNN and LSTM cells without a deep learning framework**.

## What the notebook covers

- **Vanilla RNN:** computes hidden states with `tanh` and per-step class probabilities with `softmax`; implements single-step and sequence-level backpropagation through time.
- **LSTM:** computes forget, input, and output gates with `sigmoid`, updates the cell state using a `tanh` candidate, and implements forward and backward passes across sequences.
- **Gradient tracking:** caches intermediate states and accumulates input, recurrent-weight, and bias gradients across time steps.

All computations use **Python and NumPy**. The notebook is self-contained and does not require the original course utilities or image assets.

## Run

Install NumPy and Jupyter, then open `RNN_LSTM_From_Scratch.ipynb` and run the cells in order. The notebook defines reusable functions for forward and backward computations; it does **not** include a dataset, training loop, or reported accuracy. Backward functions take upstream hidden-state gradients as input rather than differentiating an output loss.

## Source and changes

Adapted from the *Building a Recurrent Neural Network — Step by Step* notebook in DeepLearning.AI's **Sequence Models** course. The project version removes assignment prompts and grading examples, replaces the external utility import with NumPy-based activation functions, and fixes LSTM state storage and gradient propagation between time steps. The underlying RNN and LSTM equations are retained.
