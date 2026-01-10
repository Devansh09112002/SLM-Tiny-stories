import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse

# Import models and data
from src.models.transformer import BaselineTransformer
from src.models.contrastive_transformer import ContrastiveTransformer
from src.models.compressor import Compressor
from src.models.linear_model import LinearModel
from src.data.dataset import TinyStoriesDataModule
from configs.config import config

def main(args):
    
    # Setup data
    
    data_module = TinyStoriesDataModule(
        batch_size=config.data.batch_size,
        max_length=config.data.max_length,
        num_workers=config.data.num_workers
    )
    data_module.setup() # This will build the tokenizer
    
    # Update config with dynamic vocab size and special tokens
    config.model.vocab_size = data_module.tokenizer.get_vocab_size()
    config.model.pad_token_id = data_module.tokenizer.token_to_id("[PAD]")
    
    # For contrastive model, set special token IDs
    if args.model_type == 'contrastive':
        eos_token_id = data_module.tokenizer.token_to_id("[EOS]")
        # Try to find a sentence end token, fallback to period if not found
        sentence_end_token_id = data_module.tokenizer.token_to_id(".")
        if sentence_end_token_id is None:
            sentence_end_token_id = eos_token_id  # Fallback

    # Select model based on argument
    
    if args.model_type == 'transformer':
        model = BaselineTransformer(config.model, config.training)
    elif args.model_type == 'contrastive':
        model = ContrastiveTransformer(config.model, config.training)
        # Set special token IDs for contrastive model
        model.sentence_end_token_id = sentence_end_token_id
        model.eos_token_id = eos_token_id
    elif args.model_type == 'compressor':
        model = Compressor(config.model, config.training)
    elif args.model_type == 'linear':
        model = LinearModel(config.model, config.training)
    else:
        raise ValueError("Invalid model type specified")

    # Setup Trainer
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"checkpoints/{args.model_type}/",
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
    )
    
    trainer = L.Trainer(
        max_epochs=config.training.epochs,
        callbacks=[checkpoint_callback],
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        accelerator="auto",
        devices="auto"
    )

    print(f"--- Starting training for {args.model_type} model ---")
    trainer.fit(model, data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TinyStories Model")
    parser.add_argument(
        '--model_type', 
        type=str, 
        required=True,
        choices=['transformer', 'contrastive', 'compressor', 'linear'],
        help="Type of model to train."
    )
    args = parser.parse_args()
    main(args)