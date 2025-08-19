# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch
#
# Load like the following, may need depending on policy:
# python3 -m venv ~/myenv
# source ~/myenv/bin/activate
#
# Also modifications - Berlin Brown to be very verbose with output from Chatgpt, github copilot
#
# https://github.com/rasbt/LLMs-from-scratch/tree/main/ch05

import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken
import sys


# Import from local files
from previous_chapters import GPTModel, create_dataloader_v1, generate_text_simple

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    # Training Loop Explanation
    #. Gradient Reset: optimizer.zero_grad() clears any previously computed gradients. 
    # This is necessary because PyTorch accumulates gradients by default, and we want to start fresh for each batch.
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


# Wrapper function
def main(gpt_config, settings):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##############################
    # Download data if necessary
    ##############################

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    # How does this come into play?
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    # In this section, what is point of start context
    '''
    Chat conversation here
    This code snippet is calling the train_model_simple function to train a neural 
     language model. The function call initializes the training process with several 
     key parameters and captures three return values:

    The function receives the model architecture, data loaders for both training 
    and validation datasets, an optimizer that will update the model parameters, 
    and the device (CPU or GPU) on which to perform computations.

    The training configuration is specified through several parameters: num_epochs 
    (from the settings dictionary) determines how many complete passes through the 
    training data will be performed; eval_freq=5 indicates that evaluation should 
    occur every 5 training steps; and eval_iter=1 means that only 1 batch from each 
    loader will be used during evaluation for efficiency.

    The parameter start_context="Every effort moves you" provides a text prompt 
    that will be used at the end of each epoch to generate a sample output from the model. This helps monitor the model's progress qualitatively by seeing 
    how its text generation capabilities evolve during training.

    The function returns three lists which are captured in variables: train_losses 
    and val_losses track the model's performance on training and validation sets over time, 
    while tokens_seen records the cumulative number of tokens processed at each evaluation point. 
    These metrics are valuable for analyzing learning progress, detecting overfitting, 
    and understanding the model's behavior at different stages of training.
    '''
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    # Return all data tuple
    return train_losses, val_losses, tokens_seen, model

if __name__ == "__main__":

    # What does this do GPT_CONFIG_124M
    '''
    This is from GitHub - Copilot:

    Is a configuration dictionary that defines the architecture 
    and hyperparameters for a 124 million parameter GPT model. 
    This dictionary is used to initialize the GPTModel class from.

    The dictionary contains the following key parameters:

    "vocab_size": 50257 - Defines the size of the vocabulary 
       (number of unique tokens the model can process). 
       This matches the GPT-2 vocabulary size.

    "context_length": 256 - Sets the maximum sequence 
      length the model can handle (reduced from the original 1024 tokens to save memory and computation).

    "emb_dim": 768 - Specifies the embedding dimension (vector size) 
         used to represent each token.

    "n_heads": 12 - Sets the number of attention heads in each transformer block.

    "n_layers": 12 - Defines how many transformer blocks are stacked in the model.

    "drop_rate": 0.1 - Sets the dropout probability used for regularization.

    "qkv_bias": False - Indicates whether bias terms should be used in the query, key, 
       and value projections.
    '''

    '''
     The model has approximately 124 million parameters (hence the name) primarily from:

     Position Embeddings: context_length × emb_dim = 256 × 768 parameters

     Transformer Layers: 12 layers, each with attention heads and feed-forward networks

     Even though "The Verdict" is only 3,600 words:

     The model is designed to learn general language patterns, not just memorize the text
     
     The vocabulary size (50,257 tokens) matches GPT-2's full vocabulary
     
     The architecture follows smaller GPT-2 specifications (768-dim embeddings, 12 layers)

     This is the total number of unique tokens the model can recognize and generate
    '''

    '''
    The full vocabulary (50,257 tokens) gives it the capacity to represent 
    many words it never sees in training

    It's based on the GPT-2 design which was trained on millions of documents
    '''

    '''
    The word "Every" is converted to its corresponding token ID from the GPT-2 vocabulary.
    Embedding Layer: The model has an embedding layer for all 50,257 tokens, including "Every"
    '''
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
        "qkv_bias": False       # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    print("<zerlin> -- Running main routine")

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # OpenAI Released GPT-2 Under the MIT License
    # Licensing
    # There are different components to consider
    # The GPT-2 Vocabulary/Tokenizer: Free to use under MIT license
    # Your Model Architecture: Based on the GPT design but implemented by you
    # The Training Data ("The Verdict"): A separate text that may have its own copyright

    print("<zerlin> -- enter plot losses, save pdf")

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    print("<zerlin> -- save model, reload")

    # The resulting model can generate text that mimics "The Verdict" style

    '''
    Training a Model From Scratch: Creates a new language model with 
    random weights and trains it on "The Verdict" text
    '''

    # Will only generate text in the style of that specific story
    # The architecture and tokenization come from GPT-2
    # The training data and learned patterns come only from "The Verdict"

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    # Reuse the model we just saved for a basic chat bot
    # Against the model

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    context_size = GPT_CONFIG_124M["context_length"]

    print("\n<zerlin> -- Chat interface ready. Type your prompt and press Enter (type 'exit' to quit):\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("Exiting chat.")
            break

        # Encode user input
        encoded = torch.tensor(tokenizer.encode(user_input)).unsqueeze(0).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model,
                idx=encoded,
                max_new_tokens=50,
                context_size=context_size
            )
            response = tokenizer.decode(token_ids.squeeze(0).tolist())
            print("Bot:", response.replace("\n", " "))

# End of script