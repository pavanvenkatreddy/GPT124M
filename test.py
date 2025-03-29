import torch
import torch.nn.functional as F
import tiktoken
from GPT_Modules import GPT, GPTConfig  # Import your model classes

# Set up tokenizer and device
enc = tiktoken.get_encoding("gpt2")
device = torch.device("cpu")  # Explicitly set to CPU

# Load the checkpoint
checkpoint_path = "/Users/pavanvenkatreddy/Downloads/model_19072.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Extract the model state_dict from the checkpoint
compiled_state_dict = checkpoint['model']

# Clean up the state_dict by removing the '_orig_mod.' prefix
cleaned_state_dict = {}
for key, value in compiled_state_dict.items():
    cleaned_key = key.replace('_orig_mod.', '')  # Remove the unwanted prefix
    cleaned_state_dict[cleaned_key] = value

# Load the cleaned state_dict into the model
config = checkpoint['config']  # Assuming this is a dict or GPTConfig object
model = GPT(config)  # Instantiate model with the loaded config
model.load_state_dict(cleaned_state_dict)  # Load the cleaned weights
model.to(device)  # Move model to CPU
model.eval()  # Set the model to evaluation mode

# Generate text
num_return_sequences = 4
max_length = 32
tokens = enc.encode("What is the capital of India")  # Starting prompt
tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).repeat(num_return_sequences, 1)
xgen = tokens.to(device)
sample_rng = torch.Generator(device=device)
sample_rng.manual_seed(42)

# Text generation loop
while xgen.size(1) < max_length:
    with torch.no_grad():
        # Forward pass without mixed precision (just regular float32)
        logits, loss = model(xgen)  # Forward pass (B, T, vocab_size)
    logits = logits[:, -1, :]  # Take logits at the last token (B, vocab_size)
    probs = F.softmax(logits, dim=-1)  # Get the probabilities
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Top-k sampling
    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # Sample token
    xcol = torch.gather(topk_indices, -1, ix)  # Get token index
    xgen = torch.cat((xgen, xcol), dim=1)  # Append token to generated sequence

# Decode and print the generated text
for i in range(num_return_sequences):
    tokens = xgen[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(f"Sample {i}: {decoded}")
