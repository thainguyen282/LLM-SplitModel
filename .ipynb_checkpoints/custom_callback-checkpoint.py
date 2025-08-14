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