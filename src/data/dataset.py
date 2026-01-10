import os
import pytorch_lightning as L
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader, Dataset
import torch
from configs.config import config

class TinyStoriesDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        item = self.tokenized_data[idx]
        input_ids = torch.tensor(item['input_ids'])
        attention_mask = torch.tensor(item['attention_mask'])
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class TinyStoriesDataModule(L.LightningDataModule):
    def __init__(self, batch_size, max_length, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None

    def prepare_data(self):
        load_dataset(config.data.name, cache_dir=config.data.cache_dir)
    
    def _get_tokenizer(self):
        tokenizer_path = os.path.join(config.data.cache_dir, "tinystories_tokenizer.json")
        if os.path.exists(tokenizer_path):
            return Tokenizer.from_file(tokenizer_path)

        print("Tokenizer not found. Training a new one...")
        dataset = load_dataset(config.data.name, split='train', cache_dir=config.data.cache_dir)
        
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=config.model.vocab_size,
            special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        )

        def batch_iterator(batch_size=1000):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i : i + batch_size]["text"]

        tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
        tokenizer.save(tokenizer_path)
        return tokenizer

    def setup(self, stage=None):
        self.tokenizer = self._get_tokenizer()
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), length=self.max_length)
        self.tokenizer.enable_truncation(max_length=self.max_length)

        def tokenize_function(examples):
            outputs = self.tokenizer.encode_batch(examples['text'])
            return {
                'input_ids': [enc.ids for enc in outputs],
                'attention_mask': [enc.attention_mask for enc in outputs]
            }

        dataset = load_dataset(config.data.name, cache_dir=config.data.cache_dir)
        
        # Using smaller subsets for faster experimentation
        
        train_data = dataset['train'].shuffle(seed=42).select(range(100000)) # 100k samples
        val_data = dataset['validation'].shuffle(seed=42).select(range(5000)) # 5k samples

        tokenized_train = train_data.map(tokenize_function, batched=True, remove_columns=dataset['train'].column_names)
        tokenized_val = val_data.map(tokenize_function, batched=True, remove_columns=dataset['validation'].column_names)

        self.train_dataset = TinyStoriesDataset(tokenized_train)
        self.val_dataset = TinyStoriesDataset(tokenized_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)