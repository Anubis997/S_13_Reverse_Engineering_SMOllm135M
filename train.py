import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from SMOLL2_135M_imitation_model import GQA_LLM, DataLoaderLite  # Import your model and DataLoaderLite
import time  # Import the time module
from transformers import AutoTokenizer  # Import AutoTokenizer
import os  # Import os for file path handling
import torch.nn.functional as F

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create checkpoints directory if it doesn't exist
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")

# Model initialization
vocab_size = tokenizer.vocab_size  # Use the tokenizer's vocab size
dim = 768
n_heads = 12
num_kv_heads = 4
hidden_dim = 768
num_layers = 30

model = GQA_LLM(vocab_size=vocab_size, dim=dim, n_heads=n_heads, num_kv_heads=num_kv_heads, hidden_dim=hidden_dim, num_layers=num_layers)
model.to(device)

# DataLoader
B = 16  # Batch size
T = 750  # Sequence length
train_loader = DataLoaderLite(B=B, T=T)

# Training parameters
total_steps = 5000  # Total number of training steps
checkpoint_interval = 500  # Interval for generating predictions and saving checkpoints
learning_rate = 1e-4

# Initialize optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
model.train()

running_loss = 0.0

# Initialize progress bar for the first interval
progress_bar = tqdm(total=checkpoint_interval, desc=f'Steps in current interval')

# Initialize scaler for mixed precision training
scaler = torch.amp.GradScaler()

for step in range(total_steps):
    # Get batch
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)

    # Forward pass and loss computation with mixed precision
    optimizer.zero_grad()
    with torch.amp.autocast('cuda'):
        _, loss = model(x, targets=y)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # Update running loss and progress bar
    running_loss += loss.item()
    progress_bar.update(1)

    # Print progress and save checkpoint every `checkpoint_interval` steps
    if (step + 1) % checkpoint_interval == 0:
        avg_loss = running_loss / checkpoint_interval
        progress_bar.set_postfix({'Average Loss': f'{avg_loss:.4f}'})
        progress_bar.close()
        
        print(f'\nCompleted interval {(step + 1) // checkpoint_interval}/{total_steps // checkpoint_interval}')
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            # Set seeds for reproducibility
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)

            # Initialize input for generation
            initial_input = x[0:1].clone()  # Take first batch and clone it
            x = initial_input.clone()

            # Print the input sequence first
            print("\nInput Sequence:")
            input_text = tokenizer.decode(x[0, :].tolist(), skip_special_tokens=True)
            print(input_text)
            print("-" * 50)

            # Store the initial length
            initial_length = x.size(1)
          
            # Generate sequence
            while x.size(1) < initial_length+100:
                logits = model(x, targets=None)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # Take the last position
                probs = F.softmax(logits, dim=-1)
                # Top-k sampling
                topk_probs, topk_indices = torch.topk(probs, 20, dim=-1)
                ix = torch.multinomial(topk_probs, 1)  # Sample from top-k
                xcol = torch.gather(topk_indices, -1, ix)  # Get the token
                x = torch.cat((x, xcol), dim=1)  # Append to sequence
            # Print only the newly generated part
            print("\nGenerated Continuation:")
            # Only decode the tokens that were newly generated (after initial_length)
            generated_text = tokenizer.decode(x[0, initial_length:].tolist(), skip_special_tokens=True)
            print(generated_text)
            print("*" * 50)

        model.train()  # Set model back to training mode
        running_loss = 0.0  # Reset running loss
        progress_bar = tqdm(total=checkpoint_interval, desc=f'Steps in current interval')

# Save the final model checkpoint after training
final_checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_final.pt')
torch.save({
    'step': total_steps,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': running_loss / checkpoint_interval,
}, final_checkpoint_path)
print(f'Final checkpoint saved to {final_checkpoint_path}') 
