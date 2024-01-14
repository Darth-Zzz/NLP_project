import json

from utils.vocab import Vocab, LabelVocab
from utils.word2vec import Word2vecUtils
from utils.evaluator import Evaluator
from transformers import BertTokenizer

class Example():

    @classmethod
    def configuration(cls, root, train_path=None, word2vec_path=None):
        cls.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", cache_dir='./pretrained')
        cls.evaluator = Evaluator()
        cls.word_vocab = Vocab(padding=True, unk=True, filepath=train_path)
        cls.word2vec = Word2vecUtils(word2vec_path)
        cls.label_vocab = LabelVocab(root)

    @classmethod
    def load_dataset(cls, data_path, is_train=True):
        dataset = json.load(open(data_path, 'r'))
        examples = []
        for di, data in enumerate(dataset):
            for ui, utt in enumerate(data):
                ex = cls(utt, f'{di}-{ui}', is_train=is_train)
                examples.append(ex)
        return examples

    def __init__(self, ex: dict, did, is_train=True):
        super(Example, self).__init__()
        self.ex = ex
        self.did = did
        if is_train and False:
            self.utt = (
            ex["manual_transcript"]
            .replace("(unknown)", "")
            .replace("(side)", "")
            .replace("(dialect)", "")
            .replace("(robot)", "")
            .replace("(noise)", "")
        )
        else:
            self.utt = ex['asr_1best']
        self.slot = {}
        for label in ex['semantic']:
            act_slot = f'{label[0]}-{label[1]}'
            if len(label) == 3:
                self.slot[act_slot] = label[2]
        self.tags = ['O'] * len(self.utt)
        for slot in self.slot:
            value = self.slot[slot]
            bidx = self.utt.find(value)
            if bidx != -1:
                self.tags[bidx: bidx + len(value)] = [f'I-{slot}'] * len(value)
                self.tags[bidx] = f'B-{slot}'
        self.slotvalue = [f'{slot}-{value}' for slot, value in self.slot.items()]
        self.input_idx = [Example.word_vocab[c] for c in self.utt]

        l = Example.label_vocab
        self.tag_id = [l.convert_tag_to_idx(tag) for tag in self.tags]
        # print("=" * self.ex.__repr__().__len__())
        # print("self.ex", self.ex)
        # print("self.did", self.did)
        # print("self.utt", self.utt)
        # print("self.slot", self.slot)
        # print("self.tags", self.tags)
        # print("self.slotvalue", self.slotvalue)
        # print("self.input_idx", self.input_idx)
        # print("self.tag_id", self.tag_id)
