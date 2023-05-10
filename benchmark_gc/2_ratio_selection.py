import os, pickle
import argparse
from itertools import product
from datasets import get_dataset
from diff_pool import DiffPool
from gcn import L2XGCN
from gin import L2XGIN
from gsg import L2XGSG
from train_eval import cross_validation_with_val_set
from custom.utils import parse_boolean

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100) # 100
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--connected', type=parse_boolean, default=True, 
                    help='Get connected output or not')
args = parser.parse_args()

datasets = ['MUTAG',  'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI', 'DD', 'Yeast']
connected_flag = args.connected
ratios = [0.4, 0.5, 0.6, 0.7]

nets = [
    L2XGCN,
    L2XGIN,
    L2XGSG
]

def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')

best_config_fn = './base_configurations.pkl'
best_config_dict = pickle.load(open(best_config_fn, 'rb')) # load best base configuration

best_ratio_fn = './best_ratios.pkl'
if os.path.exists(best_ratio_fn):
	best_ratio_dict = pickle.load(open(best_ratio_fn, 'rb'))
else:
	best_ratio_dict = {}

results = []	
results_fn = './results_l2x_ratio.pkl'
if os.path.exists(results_fn):
	results_dict = pickle.load(open(results_fn, 'rb'))
else:
	results_dict = {}
	
for dataset_name, Net in product(datasets, nets):
    best_result = (float('inf'), 0, 0)  # (loss, acc, std)
    print(f'--\n{dataset_name} - {Net.__name__}')
    name_model = Net.__name__[-3:]
    if name_model == 'GSG':
        name_model = 'GraphSAGE'
    num_layers, hidden = best_config_dict[f'{dataset_name}-{name_model}']
    print(f'Configuration - Layers: {num_layers} - Hidden: {hidden}')
    
    for ratio in ratios:
        print(f'--\n{dataset_name} - {Net.__name__} - Ratio: {ratio} - Connected: {connected_flag}')
        ratio_str = str(ratio).replace('.', '')
        dataset = get_dataset(dataset_name, sparse=Net != DiffPool)
        model = Net(dataset, num_layers, hidden, connected_flag)
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
            best_ratio = ratio

        results_dict[f'{dataset_name}-{model}{ratio_str}'] = [acc, std]
        # save temporary results
        with open(results_fn, 'wb') as outfile:
            pickle.dump(results_dict, outfile)
            outfile.close()
            
    desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f}'

    print(f'Best configuration - Ratio: {best_ratio}')
    print(f'Test result - {desc}')
    
    results += [f'{dataset_name} - {model}: {desc}']
    best_ratio_dict[f'{dataset_name}-{model}_{connected_flag}'] = [num_layers, hidden, best_ratio]
        
    # save temporary results
    with open(best_ratio_fn, 'wb') as outfile:
        pickle.dump(best_ratio_dict, outfile)
        outfile.close()
    
results = '\n'.join(results)
print(f'--\n{results}')
