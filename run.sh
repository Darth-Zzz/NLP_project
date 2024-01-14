# #### General configuration ####
# arg_parser.add_argument('--dataroot', default='./data', help='root of data')
# arg_parser.add_argument('--word2vec_path', default='./word2vec-768.txt', help='path of word2vector file path')
# arg_parser.add_argument('--seed', default=999, type=int, help='Random seed')
# arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
# arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
# #### Training Hyperparams ####
# arg_parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
# arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
# arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
# #### Common Encoder Hyperparams ####
# arg_parser.add_argument('--encoder_cell', default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='root of data')
# arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
# arg_parser.add_argument('--embed_size', default=768, type=int, help='Size of word embeddings')
# arg_parser.add_argument('--hidden_size', default=512, type=int, help='hidden size')
# arg_parser.add_argument('--num_layer', default=2, type=int, help='number of layer')

# run baseline
python scripts/slu_baseline.py --device 0 --encoder_cell GRU > log/baseline_GRU.log

# run bert
python scripts/slu_bert.py --lr 3e-5 --device 0 --max_epoch 50 > log/bert.log

# run bert + denoise
python scripts/slu_bert_denoise.py --lr 1e-3 --device 0 --max_epoch 50 > log/bert_denoise.log

# run bert + crf
python scripts/slu_bert_crf.py --lr 1e-5 --device 0 --max_epoch 50 > log/bert_crf.log

# run bert + rnn + crf
python scripts/slu_bert_rnn_crf.py --lr 3e-5 --device 0 --max_epoch 50 > log/bert_rnn_crf.log

# run crf
python scripts/slu_crf.py --lr 2e-3 --device 0 --max_epoch 50 > log/crf.log

# run denoise
python scripts/slu_denoise.py --lr 1e-3 --device 0 --max_epoch 50 > log/denoise.log

