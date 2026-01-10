import torch
import random
from .transformer import BaselineTransformer

def shuffle_sentences(tensor, sentence_end_token_id, eos_token_id, pad_token_id):
    shuffled_tensor = tensor.clone()
    for i in range(tensor.shape[0]):
        row = tensor[i].tolist()
        try:
            # Find all sentence boundaries
            
            indices = [idx for idx, token in enumerate(row) if token in [sentence_end_token_id, eos_token_id]]
            if not indices:
                continue

            # Create sentence groups
            
            groups = []
            start_idx = 0
            for end_idx in indices:
                if row[end_idx] == pad_token_id: # stop at padding
                    break
                groups.append(row[start_idx : end_idx + 1])
                start_idx = end_idx + 1

            # Shuffle and reconstruct
            
            random.shuffle(groups)
            new_sequence = [token for group in groups for token in group]
            
            # Pad the rest of the sequence
            
            padding = [pad_token_id] * (len(row) - len(new_sequence))
            new_sequence.extend(padding)

            shuffled_tensor[i] = torch.tensor(new_sequence, device=tensor.device, dtype=torch.long)
        except Exception:
            continue
    return shuffled_tensor

class ContrastiveTransformer(BaselineTransformer):
    def __init__(self, model_config, training_config):
        super().__init__(model_config, training_config)
        self.save_hyperparameters()
        self.alpha = training_config.contrastive_alpha
        
        # Assuming tokenizer is available after setup
        # These will be set in the training script
        
        self.sentence_end_token_id = -1
        self.eos_token_id = -1

    def training_step(self, batch, batch_idx):
        
        # Standard loss on correct sequences
        
        normal_loss = self._calculate_loss(batch)

        # Create contrastive examples by shuffling sentences
        
        input_ids = batch["input_ids"]
        num_contrastive = max(1, input_ids.shape[0] // 3)
        
        contrastive_inputs = shuffle_sentences(
            input_ids[:num_contrastive], 
            self.sentence_end_token_id,
            self.eos_token_id,
            self.pad_token_id
        )
        
        contrastive_batch = {
            'input_ids': contrastive_inputs,
            'labels': contrastive_inputs,
            'attention_mask': (contrastive_inputs != self.pad_token_id).int()
        }

        contrastive_loss = self._calculate_loss(contrastive_batch)
        
        # Combine losses
        
        total_loss = normal_loss + self.alpha / (contrastive_loss + 0.01)
            
        self.log_dict({"train_loss": total_loss, "normal_loss": normal_loss, "contrastive_loss": contrastive_loss}, prog_bar=True)
        return total_loss