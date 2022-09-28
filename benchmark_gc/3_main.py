import os, pickle
import argparse
from itertools import product
from datasets import get_dataset
from diff_pool import DiffPool
from gcn import GCN, L2XGCN
from gin import GIN, L2XGIN
from train_eval import cross_validation_with_val_set
from custom.utils import parse_boolean

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--connected', type=parse_boolean, default=True, 
                    help='Get connected output or not')
args = parser.parse_args()


datasets = ['MUTAG',  'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'DD', 'Yeast']
datasets = ['MUTAG']
connected_flag = args.connected

nets = [
    GCN,
    GIN,
    L2XGCN,
    L2XGIN
]

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')

results = []
best_config_fn = './base_configurations.pkl'
best_ratio_fn = './best_ratios.pkl'
best_config_dict = pickle.load(open(best_config_fn, 'rb'))
best_ratio_dict = pickle.load(open(best_ratio_fn, 'rb'))

for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__}')
    dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
    name_model = Net.__name__
    if len(name_model) > 3:
        num_layers, hidden, ratio = best_ratio_dict[f'{dataset_name}-{name_model}']
        model = Net(dataset, num_layers, hidden, connected_flag)
    else:
        num_layers, hidden = best_config_dict[f'{dataset_name}-{name_model}']
        model = Net(dataset, num_layers, hidden)
        ratio = None
        
    print(f'--\n{dataset_name} - {Net.__name__} - Ratio: {ratio}')
    loss, acc, std = cross_validation_with_val_set(
        dataset,
        model,
        folds=10,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_step_size=args.lr_decay_step_size,
        weight_decay=0,
        ratio=ratio,
        logger=None,
    )
    if loss < best_result[0]:
        best_result = (loss, acc, std)
            
    desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'
    print(f'Best result - {desc}')
    results += [f'{dataset_name} - {model}: {desc}']

results = '\n'.join(results)
print(f'--\n{results}')
