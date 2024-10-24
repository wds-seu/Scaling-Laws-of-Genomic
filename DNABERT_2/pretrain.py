import os
import time
import torch
import random
import warnings
import argparse
import numpy as np
from dataset import load_dataset
from torch.utils.data import DataLoader
from parse import parse_method, parser_add_main_args

gpus = [5]
torch.cuda.set_device('cuda:{}'.format(gpus[0]))
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
warnings.filterwarnings('ignore')


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Parse args
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

fix_seed(args.seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load method
model = parse_method()
model = torch.nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus[0])

# print('MODEL:', model)
total = sum([param.nelement() for param in model.module.parameters()])
print('Model Size:', total)

total_step = 100000000 // args.batch_size
warmup_steps = total_step // 20


def rule(step):
    if step < warmup_steps:
        lamda = args.lr * step / warmup_steps
    else:
        lamda = args.lr * (total_step - step) / (total_step - warmup_steps)
    return lamda / args.lr


optimizer = torch.optim.AdamW([{'params': model.parameters(), 'initial_lr': args.lr}], weight_decay=args.weight_decay,
                              lr=args.lr)
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=rule, last_epoch=-1)

last_part = -1
if args.load_model is not None:
    checkpoint = torch.load(args.load_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
    last_part = checkpoint['last_part']

loss_steps = 0
for part_id in range(last_part + 1, 100):

    since_begin_part = time.time()
    trainDataset = load_dataset(part_id, is_kmer=False)
    trainLoader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model.train()
    for i, (sequence, label) in enumerate(trainLoader):
        since_begin_step = time.time()
        loss = model(**sequence, labels=label).loss.mean()
        loss /= args.accumulation_steps
        loss.backward()
        loss_steps += 1

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        warmup_scheduler.step()

        if loss_steps % args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print('\r', f'{(i + 1)}/{len(trainLoader)}, '
                        f'Loss: {args.accumulation_steps * loss.item():.8f}, '
                        f'time: {time.time() - since_begin_step:.2f}, '
                        f'LR= {lr:.9f} ', end='')

    print(f'\n Part {part_id + 1} total time: {time.time() - since_begin_part:.2f} ')

    checkpoint = {
        'last_part': part_id,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
    }
    torch.save(checkpoint, '../model/reference_sampling/230M_' + str(part_id) + '.pth')
    last_file = '../model/reference_sampling/230M_' + str(part_id - 1) + '.pth'
    if os.path.exists(last_file):
        os.remove(last_file)
