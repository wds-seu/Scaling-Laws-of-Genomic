import re
import torch
import random
import argparse
import tokenizers
import multiprocessing
from torch.utils.data import Dataset
from parse import parser_add_main_args

parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
tokenizer = tokenizers.Tokenizer.from_file("tokenizer.json")
cls_id = tokenizer.token_to_id('[CLS]')
sep_id = tokenizer.token_to_id('[SEP]')
pad_id = tokenizer.token_to_id('[PAD]')
mask_id = tokenizer.token_to_id('[MASK]')


class SequenceDataset(Dataset):
    def __init__(self):
        """
            seqs = {'input_ids': input_ids,
                   'attention_mask':attention_mask}
        """
        self.seqs = []
        self.labels = []

    def __getitem__(self, index):
        return self.seqs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


def process_line(line):
    seq = line.strip()
    seq = re.sub(r'N+', 'N', seq)  # 使用正则表达式替换连续出现的N为一个N
    seq = re.sub(r'[^ATCGN]', '', seq)  # 使用正则表达式去除除了A/T/C/G/N以外的其他碱基

    tokens = tokenizer.encode(seq)
    num_tokens = min(args.sequence_length, len(tokens.ids))
    tokens.pad(args.sequence_length, pad_id=pad_id)

    input_ids = tokens.ids
    attention_mask = tokens.attention_mask

    if len(input_ids) > args.sequence_length:
        input_ids = input_ids[:args.sequence_length - 1] + [sep_id]
        attention_mask = attention_mask[:args.sequence_length]

    # mask
    label = [-100] * args.sequence_length
    for i in range(1, num_tokens - 1):
        if random.random() < 0.15:
            if random.random() < 0.8:
                label[i] = input_ids[i]
                input_ids[i] = mask_id
            elif random.random() < 0.5:
                label[i] = input_ids[i]
                input_ids[i] = random.randint(0, tokenizer.get_vocab_size() - 1)
            else:
                label[i] = input_ids[i]
    return input_ids, attention_mask, label


def seq_to_value(seq):  # 6mer A:5/T:6/C:7/G:8/N:9/A-A-A-A-A-A:10/
    single_char_mapping = {'N': 9, 'A': 5, 'T': 6, 'C': 7, 'G': 8}
    if len(seq) == 1:
        return single_char_mapping.get(seq, -1)  # 返回单字符的值，若字符不匹配则返回 -1
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    value = 0
    for char in seq:
        value = value * 4 + mapping[char]
    return value + 10


def process_line_kmer(line):
    seq = line.strip()
    seq = re.sub(r'N+', 'N', seq)  # 使用正则表达式替换连续出现的N为一个N
    seq = re.sub(r'[^ATCGN]', '', seq)  # 使用正则表达式去除除了A/T/C/G/N以外的其他碱基

    input_ids = [cls_id]  # [CLS]
    i = 0
    while i + 5 < len(seq):
        subseq = seq[i:i + 6]  # [i,i+1,i+2,i+3,i+4,i+5]
        last_index = subseq.rfind('N')
        if last_index == -1:  # without N
            input_ids.append(seq_to_value(subseq))
            i = i + 6
        else:
            for j in range(i, i + last_index + 1):
                input_ids.append(seq_to_value(seq[j]))
            i = i + last_index + 1
    for j in range(i, len(seq)):
        input_ids.append(seq_to_value(seq[j]))
    input_ids.append(sep_id)  # [SEP]
    num_tokens = min(args.sequence_length, len(input_ids))
    attention_mask = [1] * num_tokens

    while len(input_ids) < args.sequence_length:
        input_ids.append(pad_id)
        attention_mask.append(0)

    if len(input_ids) > args.sequence_length:
        input_ids = input_ids[:args.sequence_length - 1] + [sep_id]
        attention_mask = attention_mask[:args.sequence_length]

    # mask
    label = [-100] * args.sequence_length
    for i in range(1, num_tokens - 1):
        if random.random() < 0.15:
            if random.random() < 0.8:
                label[i] = input_ids[i]
                input_ids[i] = mask_id
            elif random.random() < 0.5:
                label[i] = input_ids[i]
                input_ids[i] = random.randint(0, tokenizer.get_vocab_size() - 1)
            else:
                label[i] = input_ids[i]
    return input_ids, attention_mask, label


def load_dataset(part_id, is_kmer=False):  # 改成多进程，并且不再存储下来，每次都直接跑
    with open(args.seq_file + str(part_id) + '.txt') as f:
        lines = f.readlines()

    trainDataset = SequenceDataset()
    if is_kmer:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.map(process_line_kmer, lines)
    else:
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            results = pool.map(process_line, lines)

    for input_ids, attention_mask, label in results:
        data = {'input_ids': torch.tensor(input_ids).cpu(), 'attention_mask': torch.tensor(attention_mask).cpu()}
        label = torch.tensor(label).cpu()
        trainDataset.seqs.append(data)
        trainDataset.labels.append(label)
    return trainDataset
