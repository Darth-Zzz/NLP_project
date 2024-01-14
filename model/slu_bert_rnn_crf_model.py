#coding=utf8
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from transformers import BertTokenizer, BertModel
from torchcrf import CRF
from utils.searcher import LexiconSearcher

class SLUBertRNNCRF(nn.Module):

    def __init__(self, config):
        super(SLUBertRNNCRF, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir='./pretrained')
        self.bert = BertModel.from_pretrained("bert-base-chinese", cache_dir='./pretrained')
        self.bert.requires_grad = False
        self.config = config
        self.cell = config.encoder_cell
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=0)
        self.rnn = getattr(nn, self.cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        self.output_layer = TaggingFNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        self.searcher = LexiconSearcher('data/lexicon/poi_name.txt')

    def bert_encode(self, batch):
        utt = batch.utt
        tokenized = self.tokenizer(utt, return_tensors='pt', padding=True, truncation=True).to(self.config.device)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        token_type_ids = tokenized['token_type_ids']
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return outputs

    def forward(self, batch):
        bert_out = self.bert_encode(batch).last_hidden_state[:, 1:-1, :]
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        lengths = batch.lengths
        rnn_out, _ = self.rnn(bert_out)
        hiddens = self.dropout_layer(rnn_out)
        tag_output = self.output_layer(hiddens, tag_mask, tag_ids)

        return tag_output

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = prob[i]
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                value = self.searcher.search(slot.split("-")[1], value)
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()


class TaggingFNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TaggingFNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)
        # self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)
    def forward(self, hiddens, mask, labels=None):

        logits = self.output_layer(hiddens)
        logits += (1 - mask[:, :logits.shape[1]]).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        mask = mask.byte()
        print(mask, mask.shape)
        prob = self.crf.decode(logits, mask[:, :logits.shape[1]])
        if labels is not None:
            loss = torch.mean(-self.crf(logits, labels[:, :logits.shape[1]], mask[:, :logits.shape[1]]))
            # loss = self.loss_fct(logits.reshape(-1, logits.shape[-1]), labels[:, :logits.shape[1]].reshape(-1))
            return prob, loss
        return (prob, )
