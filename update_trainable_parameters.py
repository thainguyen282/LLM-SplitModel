import torch

def update_trainable_parameters(model, tokenizer):
    model.to(device="cuda" if torch.cuda.is_available() else "cpu",dtype=torch.bfloat16)  # Only convert device, dtype is already correct
    print(model.device)
    if model.config.is_merge:
        total_param = 0
        trainable_param = 0
        for param in model.model.layers[model.config.enc_num_layers:-model.config.dec_num_layers].parameters(): 
            param.requires_grad = False
        for param in model.middle_model.parameters(): 
            param.requires_grad = False
            total_param += param.numel()
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False
            total_param += param.numel()
        for param in model.parameters():
            if param.requires_grad:
                trainable_param += param.numel()
                total_param += param.numel()
        print(f'Total Parameters: {total_param:,}')
        print(f'Trainable Parameters: {trainable_param:,}')
        print(f'Non-trainable Parameters before: {total_param - trainable_param:,}')
        # Print percentage of trainable parameters
        print(f'Percentage of trainable parameters: {100 * trainable_param / total_param:.2f}%')
    else:
        assert False, "need to handle case is_merge = False for udpate_trainable_parameters"


    
    
    
    

    