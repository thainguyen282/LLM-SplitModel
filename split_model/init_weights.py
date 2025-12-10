import torch

def init_weights(model):
    for name, param in model.named_parameters():
        if "bias" in name:
            if "alpha_proj" in name:
                # Initialize alpha projection bias to small positive values
                # This ensures log_alpha starts with reasonable values
                torch.nn.init.constant_(param, 0.1)  # Small positive bias
            else:
                torch.nn.init.zeros_(param)
        elif "weight" in name:
            if param.dim() > 1:
                if "nvib_layer" in name and "alpha_proj" in name:
                    torch.nn.init.xavier_uniform_(param)
                    with torch.no_grad():
                        param *= 0.1
                else:
                    torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)

def weighted_mean(kl_list, weighted_mean=False):
    if weighted_mean:
        weights = [i for i in range(1, len(kl_list) + 1)]
    else: 
        weights = [1 for i in range(0, len(kl_list))]
    weights = [weight / (sum(weights)) for weight in weights]
    return sum([torch.mean(kl_layer) * weights[i] for i, kl_layer in enumerate(kl_list)])