import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer
import math
import os

print("Script is starting...")

# Try importing the model
try:
    from SMOLL2_135M_without_acceleration import LlamaForCausalLM
    print("Successfully imported LlamaForCausalLM")
except ImportError as e:
    print(f"Failed to import LlamaForCausalLM: {e}")
    exit(1)

def load_input_file_dataset(seq_length):
    """
    Loads and tokenizes text from input.txt.
    """
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("Set pad_token to eos_token")

        print("Loading input.txt file...")
        with open("input.txt", "r", encoding="utf-8") as f:
            text = f.read()

        print("Tokenizing input text...")
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=seq_length,
            return_tensors="pt",
            padding="max_length",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        print("Tokenization complete.")

        return [input_ids], tokenizer

    except Exception as e:
        print(f"Error loading input file: {e}")
        raise

def get_dataloader(dataset, batch_size):
    """
    Converts dataset into a DataLoader for efficient batching.
    """
    dataset_tensor = torch.stack(dataset)
    tensor_dataset = TensorDataset(dataset_tensor)
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

def main():
    print("Starting training script...")

    # Hyperparameters
    seq_length = 750
    batch_size = 16
    learning_rate = 0.001
    total_steps = 14000
    predict_every = 500
    checkpoint_dir = "checkpoints"
    final_checkpoint_path = "final_checkpoint.pth"

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading dataset and tokenizer...")
    dataset, tokenizer = load_input_file_dataset(seq_length)
    
    if dataset is None:
        return

    print("Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocab_size = tokenizer.vocab_size
    print(f"Final verified vocab size: {vocab_size}")

    # Model configuration
    dim = 576
    num_layers = 30
    hidden_dim = 1100
    model = LlamaForCausalLM(vocab_size, dim, num_layers, hidden_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    dataloader = get_dataloader(dataset, batch_size)
    
    print(f"\nTraining Plan:")
    print(f"Total steps: {total_steps}")
    print(f"Checkpointing every {predict_every} steps")

    # Progress bar for total steps (instead of epochs)
    step_pbar = tqdm(range(total_steps), desc="Training Progress", position=0)

    global_step = 0
    model.train()
    
    while global_step < total_steps:
        for batch in dataloader:
            if global_step >= total_steps:
                break

            input_ids = batch[0].to(device)
            labels = input_ids[:, 1:].contiguous()
            inputs = input_ids[:, :-1].contiguous()

            optimizer.zero_grad()

            # Updated autocast usage
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            step_pbar.update(1)  # Increment progress bar
            step_pbar.set_postfix({'loss': f'{loss.item():.4f}', 'step': f'{global_step}/{total_steps}'})

            # Checkpoint and text generation every 500 steps
            if global_step % predict_every == 0:
                print(f"\n{'='*80}")
                print(f"Checkpoint at Step {global_step}")
                print(f"Current loss: {loss.item():.4f}")

                # Save checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{global_step}.pth")
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

                # Generate text
                model.eval()
                with torch.no_grad():
                    sample_input = torch.randint(0, vocab_size, (1, seq_length)).to(device)
                    generated = model.generate(sample_input, max_length=10)
                    generated_text = tokenizer.decode(generated[0].cpu().numpy())
                    print(f"Generated text: {generated_text}")
                model.train()
                print(f"{'='*80}\n")
    
    step_pbar.close()
    print("\nTraining completed.")

    # Save final checkpoint after last step
    torch.save({
        'step': total_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")

if __name__ == "__main__":
    try:
        print("Entering main")
        main()
        print("Main completed")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
