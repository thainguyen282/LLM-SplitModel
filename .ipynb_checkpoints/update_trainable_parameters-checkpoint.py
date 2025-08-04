import torch

def update_trainable_parameters(model, tokenizer):
    model.to(device="cuda" if torch.cuda.is_available() else "cpu",dtype=torch.bfloat16)  # Only convert device, dtype is already correct
    # model.config.pad_token_id = tokenizer.pad_token_id
    # model.config.eos_token_id = tokenizer.eos_token_id
    # model.config.bos_token_id = tokenizer.bos_token_id
    # tokenizer.pad_token_id = tokenizer.bos_token_id
    # model.generation_config.pad_token_id = tokenizer.bos_token_id
    # model.generation_config.eos_token_id = model.config.eos_token_id
    # model.generation_config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    model.generation_config.bos_token_id = model.config.bos_token_id
    print(model.device)
    total_param = 0
    trainable_param = 0
    for param in model.parameters(): 
        param.requires_grad = False
        total_param += param.numel()
    for param in model.nvib_transformer_adapter1.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.nvib_transformer_adapter2.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.lm_head.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.model.layers[:model.config.enc_num_layers].parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.model.layers[-model.config.dec_num_layers:].parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()
    for param in model.model.norm.parameters(): 
        param.requires_grad = True
        trainable_param += param.numel()

    model.model.embed_tokens.weight.requires_grad_(True)

    print(f'Total Parameters: {total_param:,}')
    print(f'Trainable Parameters: {trainable_param:,}')
    print(f'Non-trainable Parameters: {total_param - trainable_param:,}')
    
    # Print percentage of trainable parameters
    print(f'Percentage of trainable parameters: {100 * trainable_param / total_param:.2f}%')