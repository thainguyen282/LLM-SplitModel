# LLM-SplitModel

A split model architecture for training large language models with Neural Variational Information Bottleneck (NVIB) layers. This project implements a split architecture combining Qwen2.5-Coder-7B-Instruct (base model) with Llama-3.1-8B-Instruct (middle model) for efficient code generation tasks.

## Quick Start

**For users who want to get started immediately:**

1. **Install dependencies:**
```bash
pip install torch transformers datasets peft fire wandb accelerate
huggingface-cli login  # Authenticate to access models
```

2. **Prepare your dataset** in JSON format with `instruction`, `input`, and `output` fields

3. **Run training:**
```bash
python train.py \
    --data_path="./your_dataset.json" \
    --output_dir="./output" \
    --batch_size=1 \
    --gradient_accumulation_steps=16 \
    --num_epochs=3 \
    --learning_rate=1e-4
```

4. **Monitor progress** at https://wandb.ai (after running `wandb login`)

For detailed instructions and troubleshooting, continue reading below.

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU(s) with CUDA support (recommended: A100 80Gb)
- **RAM**: 64GB+ system RAM recommended
- **Storage**: Sufficient space for models (~30GB for base models) and checkpoints

### Software Requirements

- **Python**: 3.8 or higher
- **CUDA**: 11.8 or higher (compatible with your GPU)
- **cuDNN**: Compatible with your CUDA version
- **NCCL**: For multi-GPU training (usually comes with PyTorch)

### Access Requirements

- **HuggingFace Access**: 
  - Access to `Qwen/Qwen2.5-Coder-7B-Instruct` model
  - Access to `meta-llama/Llama-3.1-8B-Instruct` model
  - Access to `iamtarun/python_code_instructions_18k_alpaca` dataset
  - You may need to request access and authenticate:
    ```bash
    huggingface-cli login
    ```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd LLM-SplitModel
```

### 2. Create a Virtual Environment

```bash
# Using conda (recommended)
conda create -n splitmodel python=3.10
conda activate splitmodel

### 3. Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies
pip install transformers>=4.35.0
pip install datasets
pip install peft
pip install fire
pip install wandb
pip install accelerate
pip install bitsandbytes  # Optional, for memory optimization

# Install additional dependencies
pip install rich  # For better console output
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

## Data Preparation

### Data Format

Your training data should be in one of the following formats:

1. **JSON/JSONL file** with the following structure:
```json
{
  "instruction": "Write a function to calculate factorial",
  "input": "",
  "output": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
}
```

2. **HuggingFace Dataset** saved to disk (using `save_to_disk()`)

### Supported Datasets

The code supports:
- Local JSON/JSONL files
- HuggingFace datasets loaded from disk
- HuggingFace datasets from the hub (e.g., `iamtarun/python_code_instructions_18k_alpaca`)

### Example: Preparing Your Dataset

```python
from datasets import Dataset, load_dataset

# Option 1: Load from JSON
dataset = load_dataset("json", data_files="your_data.json")

# Option 2: Create from list
data = [
    {"instruction": "...", "input": "...", "output": "..."},
    # ... more examples
]
dataset = Dataset.from_list(data)

# Save to disk for faster loading
dataset.save_to_disk("./my_dataset")
```

## Training

### Basic Training Command

```bash
python noloraQwen.py \
    --base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct" \
    --middle_model_path="meta-llama/Llama-3.1-8B-Instruct" \
    --data_path="/path/to/your/dataset" \
    --output_dir="./output/checkpoints" \
    --batch_size=1 \
    --gradient_accumulation_steps=16 \
    --num_epochs=3 \
    --learning_rate=1e-4 \
    --cutoff_len=4000 \
    --val_set_size=500
```

## Configuration Parameters

### Model Parameters

- `--base_model_path`: Path to the base model (default: `Qwen/Qwen2.5-Coder-7B-Instruct`)
- `--middle_model_path`: Path to the middle model (default: `meta-llama/Llama-3.1-8B-Instruct`)

### Data Parameters

- `--data_path`: Path to your dataset (JSON/JSONL file or HuggingFace dataset directory)
- `--cutoff_len`: Maximum sequence length (default: 4000)
- `--val_set_size`: Number of validation samples (default: 500)
- `--using_raw_data`: Whether to use raw data or preprocessed (default: True)
- `--prompt_template_name`: Prompt template to use (default: "alpaca")

### Training Hyperparameters

- `--batch_size`: Per-device batch size (default: 1)
- `--micro_batch_size`: Micro batch size (default: 1, usually same as batch_size)
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 16)
- `--num_epochs`: Number of training epochs (default: 3)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--warmup_steps`: Number of warmup steps (default: 750)

### Other Parameters

- `--train_on_inputs`: Whether to train on input tokens (default: True)
- `--group_by_length`: Group sequences by length for efficiency (default: False)
- `--output_dir`: Directory to save checkpoints (default: `./temp-with-Qwen-Llama-allclean-2gpu/`)

## Resuming Training

To resume from a checkpoint:

```bash
python noloraQwen.py \
    --base_model_path="Qwen/Qwen2.5-Coder-7B-Instruct" \
    --middle_model_path="meta-llama/Llama-3.1-8B-Instruct" \
    --data_path="./my_dataset" \
    --output_dir="./output" \
    --resume_from_checkpoint="./output/checkpoint-500" \
    --batch_size=2 \
    --gradient_accumulation_steps=16 \
    --num_epochs=3 \
    --learning_rate=1e-4
```

The checkpoint directory should contain:
- `pytorch_model.bin` or model files
- `optimizer.pt`
- `trainer_state.json`
- `training_args.bin`

## Monitoring Training

### Weights & Biases (wandb)

The training script automatically initializes wandb logging. To use wandb:

1. **Login to wandb (required):**
```bash
wandb login
```
   Enter your API key when prompted. You can get your API key from https://wandb.ai/authorize

2. **View logs:** Visit https://wandb.ai to view your training metrics

**Note:** The script initializes wandb with project name "split-model-with-nvib" by default. You can override this using the `--wandb_project` parameter.

### Training Outputs

The training script will:
- Print training progress every `logging_steps` (default: 10)
- Save checkpoints every `save_steps` (default: 500)
- Evaluate on validation set every `eval_steps` (default: 500)
- Log metrics including:
  - Training loss
  - Validation loss
  - KL divergence losses (KLD and KLG)
  - Learning rate
  - Training step

### Checkpoint Structure

Checkpoints are saved in `output_dir` with the following structure:
```
output/
├── checkpoint-500/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── optimizer.pt
│   ├── trainer_state.json
│   └── ...
├── checkpoint-1000/
│   └── ...
└── ...
```

## Expected Training Time

Approximate training times (varies based on hardware and dataset size):

- **Single GPU (A100 40GB)**: ~1-2 days for 3 epochs on 100k samples
## Model Architecture

The split model architecture consists of:

1. **Encoder layers** (first `enc_num_layers` from base model)
2. **NVIB adapter 1** (compression layer)
3. **Middle model layers** (if `is_merge=True`)
4. **NVIB adapter 2** (decompression layer)
5. **Decoder layers** (last `dec_num_layers` from base model)

Only specific layers are trainable:
- NVIB adapter layers
- Encoder layers (first few)
- Decoder layers (last few)
- Language model head

## 📚 Citation

### Qwen2.5
```bibtex
@article{qwen25,
  title   = {Qwen2.5 Technical Report},
  author  = {Qwen Team},
  year    = {2024},
  journal = {arXiv preprint arXiv:2407.xxxxx}
}
```
### Llama 3.1
```bibtex
@article{llama31,
  title   = {Llama 3.1: Open Foundation and Large Language Models},
  author  = {AI@Meta},
  year    = {2024},
  journal = {arXiv preprint arXiv:2407.xxxxx}
}
```
### NVIB (Neural Variational Information Bottleneck)
```bibtex
@inproceedings{alemi2017dvib,
  title     = {A VAE FOR TRANSFORMERS WITH NONPARAMETRICVARIATIONAL INFORMATION BOTTLENECK},
  author    = {James Henderson, Fabio Fehr},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017}
}
```
