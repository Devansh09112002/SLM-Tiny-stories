import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L

class LinearModel(L.LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.save_hyperparameters()

        lin_params = model_config.linear
        self.T = lin_params['context_length']
        self.d = lin_params['embedding_dim']
        self.num_tokens = model_config.vocab_size
        self.pad_token_id = model_config.pad_token_id
        self.learning_rate = training_config.learning_rate
        
        self.in_embedding = nn.Embedding(self.num_tokens, self.d)
        self.linear1 = nn.Linear(self.T * self.d, self.T * self.d)
        
        self.activation = nn.Identity()
        if lin_params['use_relu']:
            self.linear2 = nn.Linear(self.T * self.d, self.T * self.d)
            self.activation = nn.ReLU()
            
        self.out_linear = nn.Linear(self.d, self.num_tokens)
        
        mask = torch.tril(torch.ones(self.T, self.T))
        self.register_buffer('mask', mask)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

    def forward(self, x):
        B, T = x.shape
        x = self.in_embedding(x)
        x_flat = x.view(B, T * self.d)
        
        # Apply mask to the weight of the linear layer
        
        w1 = self.linear1.weight.view(self.T, self.d, self.T, self.d)
        w1 = w1 * self.mask[None, :, None, :]
        w1 = w1.view(self.T * self.d, self.T * self.d)
        x_flat = F.linear(x_flat, w1, self.linear1.bias)
        
        x = x_flat.view(B, T, self.d)
        x = self.activation(x)
        
        if hasattr(self, 'linear2'):
            w2 = self.linear2.weight.view(self.T, self.d, self.T, self.d)
            w2 = w2 * self.mask[None, :, None, :]
            w2 = w2.view(self.T * self.d, self.T * self.d)
            x_flat = x.view(B, T*self.d)
            x_flat = F.linear(x_flat, w2, self.linear2.bias)
            x = x_flat.view(B, T, self.d)

        return self.out_linear(x)

    def _calculate_loss(self, batch):
        src = batch['input_ids']
        tgt = batch['labels']
        
        # Linear model expects fixed context length
        
        src = src[:, :self.T]
        tgt = tgt[:, :self.T]
        
        logits = self(src)
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = tgt[..., 1:].contiguous()
        
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)