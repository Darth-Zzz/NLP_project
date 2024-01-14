# coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertTokenizer, BertModel


class SLUBertDenoise(nn.Module):
    def __init__(self, config):
        super(SLUBertDenoise, self).__init__()
        self.config = config
        print("vocab_size", config.vocab_size)
        print("embed_size", config.embed_size)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir='./pretrained')
        self.bert = BertModel.from_pretrained("bert-base-chinese", cache_dir='./pretrained')
        # self.bert.requires_grad = False
        self.cell = config.encoder_cell
        #self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        #self.rnn = getattr(nn, self.cell)(config.embed_size,config.hidden_size // 2,num_layers=config.num_layer,bidirectional=True,batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(768, config.num_tags, config.tag_pad_idx)
        self.denoise_layer = getattr(nn, self.cell)(config.embed_size, config.embed_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.loss = nn.MSELoss()

    def bert_encode(self, utt, pad_length=None):

        tokenized = self.tokenizer(utt, return_tensors='pt', padding='max_length', truncation=True, max_length=pad_length).to(self.config.device)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        token_type_ids = tokenized['token_type_ids']
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs

    def forward(self, batch):
        pad_length = max([len(batch.utt), len(batch.denoise_utt)])
        bert_out = self.bert_encode(batch.utt, pad_length).last_hidden_state[:, 1:-1, :]
        bert_out_gt = self.bert_encode(batch.denoise_utt, pad_length).last_hidden_state[:, 1:-1, :]
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        lengths = batch.lengths
        
        #denoise_inputs = rnn_utils.pack_padded_sequence(bert_out, lengths, batch_first=True, enforce_sorted=False)
        denoise_rnn_out, _ = self.denoise_layer(bert_out)
        #denoise_out, denoise_len = rnn_utils.pad_packed_sequence(denoise_rnn_out, batch_first=True)
        
        denoise_loss = self.loss(bert_out_gt, denoise_rnn_out)
        
        hiddens = self.dropout_layer(bert_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)
        return tag_output[0], tag_output[1]+denoise_loss

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        prob, loss = self.forward(batch)
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[: len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == "O" or tag.startswith("B")) and len(tag_buff) > 0:
                    slot = "-".join(tag_buff[0].split("-")[1:])
                    value = "".join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f"{slot}-{value}")
                    if tag.startswith("B"):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith("I") or tag.startswith("B"):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = "-".join(tag_buff[0].split("-")[1:])
                value = "".join([batch.utt[i][j] for j in idx_buff])
                pred_tuple.append(f"{slot}-{value}")
            predictions.append(pred_tuple)
        return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        # logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            truncate_length = min([logits.shape[1], labels.shape[1]])
            loss = self.loss_fct(logits[:, :truncate_length].reshape(-1, logits.shape[-1]), labels[:, :truncate_length].reshape(-1))
            return prob, loss
        return (prob, )
