import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
import math

class Encoder(nn.Module):
    def __init__(
        self,
        n_positions,
        vocab_size,
        d_model,
        n_head,
        dim_feedforward,
        n_layers,
        dropout,
        compression_ratio,
        pad_token_id=1000,
        bos_token_id=1001,
        cls_token_id=1002,
    ):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id

        self.compression_ratio = compression_ratio
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(n_positions, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, n_head, dim_feedforward, dropout, batch_first=True
            ),
            n_layers,
        )
        if compression_ratio != 1:
            self.downscale = nn.Linear(d_model * 2, int(d_model * 2 // compression_ratio))

    def forward(self, src):
        b, s = src.shape  # [batch_size, sequence_length]
        chunks = src.reshape(b * (s // 2), 2)
        chunks = torch.cat(
            [
                chunks,
                torch.full((chunks.shape[0], 1), self.cls_token_id, device=src.device),
            ],
            dim=1,
        )

        position_ids = torch.arange(s, device=src.device).unsqueeze(0).expand(b, s)
        chunked_position_ids = position_ids.reshape(b * (s // 2), 2)

        positional_embeddings = self.position_embedding(chunked_position_ids)
        positional_embeddings = torch.cat(
            [
                positional_embeddings,
                torch.zeros(
                    (positional_embeddings.shape[0], 1, self.d_model), device=src.device
                ),
            ],
            dim=1,
        )

        embeddings = self.embedding(chunks) + positional_embeddings
        padding_mask = (chunks == self.pad_token_id).float()

        h = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        h = h[:, :-1, :].reshape(b, s // 2, self.d_model * 2)

        if self.compression_ratio != 1:
            h = self.downscale(h)

        padding_mask = padding_mask[:, 0].reshape(b, s // 2)
        return h, padding_mask


class TrunkTransformer(nn.Module):
    def __init__(
        self, d_model, n_head, dim_feedforward, n_layers, dropout, compression_ratio
    ):
        super(TrunkTransformer, self).__init__()
        self.compression_ratio = compression_ratio
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                int(d_model * 2 // compression_ratio),
                n_head,
                dim_feedforward,
                dropout,
                batch_first=True,
            ),
            n_layers,
        )

    def forward(self, src, mask=None, padding_mask=None):
        return self.transformer(src, mask, padding_mask, is_causal=True)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_positions,
        d_model,
        n_head,
        dim_feedforward,
        n_layers,
        dropout,
        compression_ratio,
        pad_token_id=1000,
        bos_token_id=1001,
    ):
        super(Decoder, self).__init__()
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

        self.d_model = int(d_model * 2 // compression_ratio)
        self.compression_ratio = compression_ratio
        self.embedding = nn.Embedding(vocab_size, self.d_model)
        self.position_embedding = nn.Embedding(n_positions, self.d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                self.d_model, n_head, dim_feedforward, dropout, batch_first=True
            ),
            n_layers,
        )

    def forward(self, src, tgt):
        b, s = tgt.shape  # [batch_size, sequence_length]
        tgt = tgt.reshape(-1, 2)
        tgt = torch.cat(
            [
                torch.full((tgt.shape[0], 1), self.bos_token_id, device=src.device),
                tgt[:, :-1],
            ],
            dim=1,
        )

        tgt_key_padding_mask = (tgt == self.pad_token_id).float()

        position_ids = torch.arange(s, device=tgt.device).unsqueeze(0).expand(b, s)
        chunked_position_ids = position_ids.reshape(-1, 2)
        # Take only the first element of each pair for position embedding
        chunked_position_ids = chunked_position_ids[:, :1].squeeze(1)

        positional_embeddings = self.position_embedding(chunked_position_ids).unsqueeze(1)
        positional_embeddings = torch.cat(
            [
                torch.zeros(
                    (
                        positional_embeddings.shape[0],
                        1,
                        positional_embeddings.shape[-1],
                    ),
                    device=src.device,
                ),
                positional_embeddings,
            ],
            dim=1,
        )

        tgt = self.embedding(tgt) + positional_embeddings
        src = src.reshape(-1, 1, src.shape[-1])
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.shape[1], device=src.device
        )

        return self.transformer(
            tgt,
            src,
            tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=causal_mask,
            tgt_is_causal=True,
        )


# Main Lightning Module
class Compressor(L.LightningModule):
    def __init__(self, model_config, training_config):
        super().__init__()
        self.save_hyperparameters()
        self.pad_token_id = model_config.pad_token_id
        self.bos_token_id = getattr(model_config, 'bos_token_id', 1)  # Default if not set
        self.learning_rate = training_config.learning_rate

        comp_params = model_config.compressor
        self.compression_ratio = comp_params['compression_ratio']

        self.compressor = Encoder(
            n_positions=comp_params['n_positions'],
            vocab_size=model_config.vocab_size,
            d_model=comp_params['d_model'],
            n_head=comp_params['n_head'],
            dim_feedforward=comp_params['dim_feedforward'],
            n_layers=comp_params['encoder_layers'],
            dropout=comp_params['dropout'],
            compression_ratio=self.compression_ratio,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            cls_token_id=getattr(model_config, 'cls_token_id', 2)  # Default CLS token ID
        )
        self.trunk = TrunkTransformer(
            d_model=comp_params['d_model'],
            n_head=comp_params['n_head'],
            dim_feedforward=comp_params['dim_feedforward'],
            n_layers=comp_params['trunk_layers'],
            dropout=comp_params['dropout'],
            compression_ratio=self.compression_ratio
        )
        self.decompressor = Decoder(
            n_positions=comp_params['n_positions'],
            vocab_size=model_config.vocab_size,
            d_model=comp_params['d_model'],
            n_head=comp_params['n_head'],
            dim_feedforward=comp_params['dim_feedforward'],
            n_layers=comp_params['decoder_layers'],
            dropout=comp_params['dropout'],
            compression_ratio=self.compression_ratio,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id
        )
        self.linear = nn.Linear(
            int(comp_params["d_model"] * 2 // self.compression_ratio),
            model_config.vocab_size
        )

    def forward(self, src, tgt):
        h, padding_mask = self.compressor(src)
        mask = nn.Transformer.generate_square_subsequent_mask(h.shape[1], device=src.device)
        h = self.trunk(h, mask, padding_mask)
        h = self.decompressor(h, tgt)
        return self.linear(h.reshape(src.shape[0], -1, h.shape[-1]))

    def _common_step(self, batch, batch_idx):
        input_ids, labels = batch["input_ids"], batch["labels"]
        # Compressor expects sequence length to be even
        seq_len = input_ids.shape[1]
        if seq_len % 2 != 0:
            input_ids = input_ids[:, :-1]
            labels = labels[:, :-1]
        
        input_ids_shifted, labels_shifted = input_ids[:, :-2], labels[:, 2:]
        output = self(input_ids_shifted, labels_shifted)
        loss = F.cross_entropy(
            output.reshape(-1, output.size(-1)),
            labels_shifted.reshape(-1),
            ignore_index=self.pad_token_id,
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)