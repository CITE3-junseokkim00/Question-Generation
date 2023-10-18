import argparse
import numpy as np
import pandas as pd
import torch
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from dataset import QuestionGeneratorModule
from model import KoBARTQuestionGenerator

from transformers import PreTrainedTokenizerFast

parser = argparse.ArgumentParser()




if __name__=='__main__':
    parser.add_argument('--train_data_dir', type=str, default='./data/korquad_1.0/train.csv')
    parser.add_argument('--test_data_dir', type=str, default='./data/korquad_1.0/test.csv')
    parser.add_argument('--batch_size', type=int, default= 8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint',type=str,default='./checkpoint2')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)

    args = parser.parse_args()
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')


    dm = QuestionGeneratorModule(train_data_dir=args.train_data_dir,
                                 test_data_dir=args.test_data_dir,
                                 tokenizer=tokenizer,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers)
    model = KoBARTQuestionGenerator(lr=args.lr)
    wandb_logger=WandbLogger(project="Question Generation using KoBart")
    checkpoint_callback= ModelCheckpoint(monitor='val_loss', 
                                         dirpath=args.checkpoint,
                                         verbose=True,
                                         save_last=True,
                                         mode='min',
                                         save_top_k=3)
    
    trainer = pl.Trainer(max_epochs=args.max_epochs,
                         accelerator='gpu', 
                         devices=1,
                         gradient_clip_val=args.gradient_clip_val,
                         callbacks=[checkpoint_callback],
                         logger=wandb_logger)
    trainer.fit(model, dm)
    # trainer.validate(model, dm)
    trainer.test(model, dm)