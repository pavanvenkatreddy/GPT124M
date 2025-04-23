# GPT124M
This repository contains my implementation of the GPT-2 architecture from scratch, inspired by Andrej Karpathy’s Neural Networks: Zero to Hero series and OpenAI’s original GPT-2 model. The goal is to deeply understand the inner workings of transformer-based language models by building each component from the ground up.

---

## Prerequisites

Before you start, make sure you have:

- A **Lambda Labs** account ([Sign up here](https://lambdalabs.com/))
- Basic familiarity with **Linux**, **Python**, and **PyTorch**
- An SSH client to connect to the instance

---

## Step 1: Set Up File System Storage

To persist your dataset and model checkpoints across multiple instances, you need a **file system** that can be attached to different nodes.

1. **Navigate to the Lambda Labs Console**
2. Go to **File Systems** and create a new storage
3. **Attach** this storage to an instance in the next step

---

## Step 2: Launch a Small Instance and Prepare the Dataset

1. **Create a new instance** (any small instance should be fine)
2. **Attach the file system** created in Step 1
3. SSH into your instance and navigate to the attached file system:
   ```bash
   cd /mnt/path-to-your-filesystem
   ```
4. **Clone this repository** inside the file system:
   ```bash
   git clone https://github.com/pavanvenkatreddy/GPT124M.git
   cd GPT124M
   ```
5. **Run the data generator script** to generate the dataset:
   ```bash
   python data_generator.py
   ```
6. **Shut down this instance** once the dataset is prepared:
   ```bash
   sudo shutdown -h now
   ```

---

## Step 3: Launch a High-Performance GPU Instance

1. **Create a new instance** with **8xA100 SXM 80GB GPUs**
   - Alternatively, you can train on a **single GPU**, but it will be significantly slower.
2. **Attach the previously created file system** to the new instance
3. SSH into the instance and navigate to the repository directory:
   ```bash
   cd /mnt/path-to-your-filesystem/GPT124M
   ```

---

## Step 4: Train the Model

Run the training script using PyTorch's distributed training:

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

This will launch training across **all 8 GPUs** using PyTorch's **distributed training** feature.

### Model Architecture

This implementation is based on **GPT-2 (124M)**, a **decoder-only Transformer** with:

- **12 layers (transformer blocks)**
- **12 attention heads per layer**
- **768-dimensional embeddings**
- **LayerNorm and GELU activation**
- **Causal self-attention with masked multi-head attention**
- **AdamW optimizer with weight decay for stable training**

---

## Step 5: Export the Model

Once training is complete, export the trained model to your preferred destination.

To save the model locally:

```python
import torch
from GPT_Modules import GPTConfig, GPT

# Load the trained model
checkpoint = torch.load("model_checkpoint.pt", map_location="cpu")
config = checkpoint["config"]
model = GPT(config)
model.load_state_dict(checkpoint["model"])
model.eval()

# Save model weights
torch.save(model.state_dict(), "trained_model.pth")
```

To upload to **Hugging Face Hub**:

```bash
pip install huggingface_hub
huggingface-cli login  # Authenticate first
```

Then, use:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="path-to-your-trained-model", repo_id="your-huggingface-username/GPT124M")
```

---

## Additional Notes

- You can experiment with **different datasets** or **fine-tune this model** using **RLHF** (Reinforcement Learning with Human Feedback).
- To resume training from a checkpoint, modify `train.py` to load weights from the latest saved model.
- The dataset (`FineWeb-Edu10B`) is a subset of **FineWeb**, filtered to focus on educational and high-quality text.

---

## Resources

- [Lambda Labs Documentation](https://lambdalabs.com/)
- [Hugging Face Model Upload Guide](https://huggingface.co/docs/hub/en/upload)
- [Andrej Karpathy’s Zero to Hero Series](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

