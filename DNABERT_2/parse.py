from bert_layers import *
from configuration_bert import *


def parse_method():
    config = BertConfig().from_json_file('config.json')
    model = BertForMaskedLM(config)
    return model


def parser_add_main_args(parser):
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seq_file', type=str, default='../gene_library_construction/variant_sampling/')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--sequence_length', type=int, default=128)
    parser.add_argument('--accumulation_steps', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--load_model', type=str, default=None)
    # parser.add_argument('--load_model', type=str, default='../model/variant_sampling/117M_90.pth')
