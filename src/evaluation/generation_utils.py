import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, input_ids, max_new_tokens=50, temperature=0.7, top_k=50):
    """
    Generate text completion for any of the models.
    
    Args:
        model: The trained model
        tokenizer: The tokenizer
        input_ids: Input token IDs
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
    
    Returns:
        output_ids: Generated token IDs
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    generated_ids = input_ids.clone()
    pad_token_id = tokenizer.token_to_id("[PAD]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get model predictions based on model type
            try:
                if hasattr(model, 'pad_token_id'):
                    # For transformer-based models (transformer, contrastive)
                    padding_mask = (generated_ids == model.pad_token_id)
                    logits = model(generated_ids, padding_mask)
                elif 'compressor' in str(type(model)).lower():
                    # For compressor model - needs special handling
                    # Compressor expects input and target, use same sequence for generation
                    if generated_ids.shape[1] % 2 != 0:
                        # Pad to even length for compressor
                        pad_tensor = torch.full((generated_ids.shape[0], 1), pad_token_id, device=generated_ids.device)
                        generated_ids = torch.cat([generated_ids, pad_tensor], dim=1)
                    
                    input_seq = generated_ids[:, :-2] if generated_ids.shape[1] > 2 else generated_ids
                    target_seq = generated_ids[:, 2:] if generated_ids.shape[1] > 2 else generated_ids
                    logits = model(input_seq, target_seq)
                else:
                    # For linear model and others
                    logits = model(generated_ids)
            except Exception as e:
                print(f"Error in model forward pass: {e}")
                # Fallback: just return the input
                return generated_ids
            
            # Get logits for the last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == eos_token_id:
                break
    
    return generated_ids