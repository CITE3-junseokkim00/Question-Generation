import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import lightning.pytorch as pl


class QGDataset(Dataset):
    def __init__(self, file, tokenizer, max_len=512, ignore_index=-100, shuffle=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.file = pd.read_csv(file)
        if shuffle:
            self.file.sample(frac=1)
        self.len = self.file.shape[0]
        self.sep_token = '<unused0>'
        self.pad_index = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index
    
    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
        return inputs

    def add_ignore_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]

        return inputs

    def __getitem__(self, index):
        context, question, answer = self.file['context'].iloc[index], self.file['question'].iloc[index], self.file['answers'].iloc[index]
        input_ids = self.tokenizer.encode(context+self.sep_token+answer)
        input_ids = self.add_padding_data(input_ids)

        label_ids = self.tokenizer.encode(question)
        label_ids.append(self.tokenizer.eos_token_id)
        

        dec_input_ids = [self.tokenizer.bos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)
        label_ids = self.add_ignore_data(label_ids)

        return {
            'input_ids': np.array(input_ids, dtype=np.int_),
            'decoder_input_ids': np.array(dec_input_ids, dtype=np.int_),
            'label_ids': np.array(label_ids, dtype=np.int_)
        }

    def __len__(self):
        return self.len
    


class QuestionGeneratorModule(pl.LightningDataModule):
    def __init__(self, train_data_dir, test_data_dir, tokenizer, batch_size, num_workers):
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        pass

    def prepare_data(self):
        # Download data
        pass

    def setup(self, stage):
        #multiple gpu
        dataset = QGDataset(file=self.train_data_dir,tokenizer=self.tokenizer, shuffle=True)
        self.train, self.val = random_split(dataset=dataset, lengths=[0.8, 0.2])
        self.test = QGDataset(file=self.test_data_dir, tokenizer=self.tokenizer)
        

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(dataset=self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(dataset=self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
