import streamlit as st
import torch
import tiktoken
from GPT_Modules import GPT, GPTConfig  # Import your GPT model classes
from huggingface_hub import hf_hub_download

# Replace with the repository and file path
repo_id = "pavanvenkat/GPT124M"
file_name = "model_19072.pt"  # The specific file you want to download
file_path = hf_hub_download(repo_id=repo_id, filename=file_name,)
print(f"File downloaded to: {file_path}")

# Set up device and tokenizer
device = torch.device("cpu")  # Use "cuda" if you have a GPU
enc = tiktoken.get_encoding("gpt2")

# Load your trained GPT model
checkpoint_path = file_path  # Adjust this to your model's path
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

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

# Streamlit app UI
st.title("GPT124M")
st.write("Enter a prompt to generate text:")

user_input = st.text_area("Your Prompt:", "Once upon a time")

if st.button("Generate"):
    tokens = enc.encode(user_input)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    
    num_return_sequences = 1
    max_length = 100

    xgen = tokens
    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)

    # Generate text
    with torch.no_grad():
        while xgen.size(1) < max_length:
            logits, _ = model(xgen)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)
            xcol = torch.gather(topk_indices, -1, ix)
            xgen = torch.cat((xgen, xcol), dim=1)

    generated_text = enc.decode(xgen[0, :max_length].tolist())
    st.subheader("Generated Text:")
    st.write(generated_text)
