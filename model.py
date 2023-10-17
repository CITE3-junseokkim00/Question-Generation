import pandas as pd
import lightning.pytorch as pl
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
import torchmetrics
import torch


class KoBARTQuestionGenerator(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained('gogamza/kobart-base-v2')
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
        self.lr = lr
        self.train_loss=[]
        self.val_loss=[]
        self.test_loss=[]
    
    def forward(self, input_ids, decoder_input_ids, labels):
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).float()
        decoder_attention_mask = decoder_input_ids.ne(self.tokenizer.pad_token_id).float()

        return self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          decoder_input_ids=decoder_input_ids,
                          decoder_attention_mask=decoder_attention_mask,
                          labels=labels, return_dict=True)
    # def forward(self, inputs):
    #     attention_mask = inputs['input_ids'].ne(self.tokenizer.pad_token_id).float()
    #     decoder_attention_mask = inputs['decoder_input_ids'].ne(self.tokenizer.pad_token_id).float()

    #     return self.model(input_ids=inputs['input_ids'],
    #                       attention_mask=attention_mask,
    #                       decoder_input_ids=inputs['decoder_input_ids'],
    #                       decoder_attention_mask=decoder_attention_mask,
    #                       labels=inputs['labels'], return_dict=True)
    
    def training_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels = batch['input_ids'], batch['decoder_input_ids'], batch['label_ids']
        output = self.forward(input_ids=input_ids,decoder_input_ids=decoder_input_ids,labels=labels)
        self.train_loss.append(output.loss)
        self.log('train_loss', output.loss)
        return output.loss

    def on_training_epoch_end(self):
        train_hat_loss = torch.FloatTensor(self.train_loss).mean()
        self.log('mean_train_loss', train_hat_loss)
        self.train_loss.clear()

    def validation_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels = batch['input_ids'], batch['decoder_input_ids'], batch['label_ids']
        output = self.forward(input_ids=input_ids,decoder_input_ids=decoder_input_ids,labels=labels)
        self.val_loss.append(output.loss)
        self.log('val_loss', output.loss)
        return output.loss
    
    def on_validation_epoch_end(self):
        val_hat_loss = torch.FloatTensor(self.val_loss).mean()
        self.log('mean_val_loss', val_hat_loss)
        self.val_loss.clear()

    def test_step(self, batch, batch_idx):
        input_ids, decoder_input_ids, labels = batch['input_ids'], batch['decoder_input_ids'], batch['label_ids']
        output = self.forward(input_ids=input_ids,decoder_input_ids=decoder_input_ids,labels=labels)
        self.test_loss.append(output.loss)
        self.log('test_loss', output.loss)
        return output.loss
    def on_test_epoch_end(self):
        test_hat_loss = torch.FloatTensor(self.test_loss).mean()
        self.log('mean_test_loss', test_hat_loss)
        self.test_loss.clear()

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr, correct_bias=False)

        # scheduler
        scheduler = get_cosine_schedule_with_warmup(optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=self.trainer.estimated_stepping_batches,)
        
        lr_scheduler = {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [lr_scheduler]