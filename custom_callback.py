from transformers import TrainerCallback
import wandb
import gc 
import torch

class KLStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        model = kwargs['model']
        model.kl_step = state.global_step

class KLMetricsCallback(TrainerCallback):
    """Callback to log KL metrics and training statistics."""
    
    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs['model']
        
        if wandb.run is not None and hasattr(model, 'kld') and hasattr(model, 'klg'):
            try:
                # Log total KL losses if they exist
                if model.kld is not None and model.klg is not None:
                    wandb.log({
                        "kl/total_kld": model.kld.item(),
                        "kl/total_klg": model.klg.item(),
                        "kl/total_combined": (model.kld + model.klg).item(),
                        "kl/step": state.global_step
                    })
                
                # Log KL annealing information
                if hasattr(model, 'kl_annealing_scheduler') and model.kl_annealing_scheduler is not None:
                    kl_factor = model.kl_annealing_scheduler(state.global_step)
                    wandb.log({
                        "kl/annealing_factor": kl_factor,
                        "kl/annealing_step": state.global_step,
                    })
            except Exception as e:
                print(f"Warning: Could not log KL metrics in callback: {e}")

class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()
    def on_prediction_step(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

from transformers import TrainerCallback
import torch

class GenerateOnTrainExampleCallback(TrainerCallback):
    def __init__(self, tokenizer, model, train_dataset, every_n_steps=500, max_new_tokens=512):
        self.tokenizer = tokenizer
        self.model = model
        self.train_dataset = train_dataset
        self.every_n_steps = every_n_steps
        self.max_new_tokens = max_new_tokens

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only trigger every_n_steps
        if state.global_step % self.every_n_steps == 0 and state.global_step > 0:
            print(f"\n[Step {state.global_step}] Generating sample output:")

            # Take first datapoint of training set
            first_example = self.train_dataset[0]
            
            input_ids = torch.tensor(first_example["prompt"], dtype=torch.long).unsqueeze(0).to(self.model.device)

            # Decode original input (for display)
            prompt = self.tokenizer.decode(first_example["prompt"], skip_special_tokens=False)

            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_new_tokens,
                )

            generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=False)
            print(f"Prompt: {prompt}\n---\nGenerated:\n{generated}\n")

        return control