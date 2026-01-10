import torch
import torch.nn as nn
import math
import pytorch_lightning as L

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class BaselineTransformer(L.LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.save_hyperparameters()

        self.d_model = model_config.transformer['hidden_size']
        self.learning_rate = training_config.learning_rate
        self.pad_token_id = model_config.pad_token_id

        self.embedding = nn.Embedding(model_config.vocab_size, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=512)  # Use a reasonable default
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=model_config.transformer['heads'], 
            dim_feedforward=model_config.transformer['feedforward_dim'], 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=model_config.transformer['layers'])
        self.linear = nn.Linear(self.d_model, model_config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)

    def forward(self, src, src_key_padding_mask):
        src_attn_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        output = self.transformer_encoder(
            src, 
            mask=src_attn_mask, 
            src_key_padding_mask=src_key_padding_mask
        )
        return self.linear(output)

    def _calculate_loss(self, batch):
        src = batch['input_ids']
        tgt = batch['labels']
        padding_mask = (src == self.pad_token_id)
        
        logits = self(src, padding_mask)
        
        # Shift logits and labels for next-token prediction
        
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