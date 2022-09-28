import os, pickle
import argparse
from itertools import product

from datasets import get_dataset
from diff_pool import DiffPool
from gcn import GCN
from gin import GIN
from train_eval import cross_validation_with_val_set

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100) # 100
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
args = parser.parse_args()

layers = [1, 2, 3, 4]
hiddens = [16, 32, 64, 128]
datasets = ['MUTAG',  'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'DD', 'Yeast']
datasets = ['MUTAG']

nets = [
    GCN,
    GIN,
]


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')

best_config_fn = './base_configurations.pkl'
if os.path.exists(best_config_fn):
	best_config_dict = pickle.load(open(best_config_fn, 'rb'))
else:
	best_config_dict = {}
	
	
results = []	
results_fn = './results_baselines.pkl'
if os.path.exists(best_config_fn):
	results_dict = pickle.load(open(results_fn, 'rb'))
else:
	results_dict = {}

for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__}')
    name_model = Net.__name__[:3]
    for num_layers, hidden in product(layers, hiddens):
        
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(dataset, num_layers, hidden)
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
            ratio=None,
            logger=None,
        )
        if loss < best_result[0]:
            best_result = (loss, acc, std)
            best_num_layers = num_layers
            best_hidden_dim = hidden
            
        results_dict[f'{dataset_name}-{model}{num_layers}_{hidden}'] = [acc, std]
        # save temporary results
        with open(results_fn, 'wb') as outfile:
            pickle.dump(results_dict, outfile)
            outfile.close()
            
    desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'

    print(f'Best configuration - Layers: {best_num_layers} - Hidden: {best_hidden_dim}')
    print(f'Test result - {desc}')

    results += [f'{dataset_name} - {model}: {desc}']
    best_config_dict[f'{dataset_name}-{model}'] = [best_num_layers, best_hidden_dim]
    
    # save temporary results
    with open(best_config_fn, 'wb') as outfile:
        pickle.dump(best_config_dict, outfile)
        outfile.close()
        
results = '\n'.join(results)
print(f'--\n{results}')
