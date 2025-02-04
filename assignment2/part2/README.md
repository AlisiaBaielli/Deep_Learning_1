# Assignment 2, Part 2: Micro-GPT

This folder contains the code for implementing your own GPT2 model. The model is trained to predict next characters. The code is structured in the following way:

* `dataset.py`: Contains the implementation of the character-level text dataset and tokenizer. It includes:
  - `CharTokenizer`: A class that handles conversion between characters and indices, with methods for encoding text to indices and decoding indices back to text.
  - `TextDataset`: A PyTorch Dataset class that processes text data in blocks, preparing it for training the GPT model. It loads text data and provides character sequences for training, where each input sequence (x) is paired with its corresponding target sequence (y) shifted by one character.
* `gpt.py`: Contains template classes for the building block of the GPT decoder-only model.
* `cfg.py`: Contains the configuration setup using ArgumentParser. It defines various hyperparameters and settings for the model training, including:
  - Model configuration (text file path, model type, block size, pretrained options)
  - Training parameters (batch sizes, learning rate, optimizer settings, number of epochs)
  - System settings (logging directory, seed, number of workers)
  - Performance options (flash attention, precision, model compilation (torch.compile))
* `train.py`: Contains the training implementation for the GPT model using PyTorch Lightning. It handles model training, evaluation, text generation during training, and supports both training from scratch and fine-tuning pretrained models. The code uses TensorBoard for logging and includes standard training optimizations like gradient clipping and precision options.


