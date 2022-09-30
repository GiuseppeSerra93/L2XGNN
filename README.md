# L2XGNN: Learning to Explain Graph Neural Networks
The repository contains the code to reproduce the results of our [paper](https://arxiv.org/abs/2209.14402). The source code consists of two folders:
- `benchmark_gc` to reproduce the experiments related to graph classification tasks.
- `xai_evaluation` to run, train and evaluate L2XGNN in comparison with common XAI post-hoc techniques.

### Requirements
The scripts are implemented with Python 3.9, and tested with Linux OS.
 - `pyg==2.0.3` 
 - `pytorch==1.10.1`
 - `networkx==2.6.3`
 - `numpy==1.21.2`
 - `dive-into-graphs==0.2.0`
 - `scikit-learn==1.0.2`
 - `matplotlib=3.5.1` 

## Graph Classification 
The folder `benchmark_gc` contains the evaluation script for various methods on [common benchmark datasets](http://graphkernels.cs.tu-dortmund.de) via 10-fold cross validation, where a training fold is randomly sampled to serve as a validation set. We slightly modify the [evaluation protocol](https://github.com/pyg-team/pytorch_geometric/tree/master/benchmark/kernel) to work for our case.

### Base model hyperparameter selection
Before training L2XGNN, we need to find the best configuration for each base models considered (i.e., GCN and GIN). Hyperparameter selection is performed for the number of hidden units [16, 32, 64, 128] and the number of layers [1, 2, 3, 4] with respect to the validation set. First, run the following command:
 - `python 1_backbone_selection.py`

### L2XGNN hyperparameter selection and evaluation
Once the hyperparameters of the default models are found, we select the best ratio R from the set of values [0.4,  0.5,  0.6,  0.7]  based on the validation accuracy.
 - `python 2_ratio_selection.py --connected={}`
       - `connected`: parameter to decide between connected and disconnected subgraphs (default value `True`).
 
Finally, we can select the perturbation intensity `lambda_` from the list of values [10, 100, 1000]. This parameter can be manually changed in the following file `l2xgnn/graph_imle.py`. Below, the corresponding code snippet for the connected strategy:
```
class GraphIMLETopKConnected(torch.autograd.Function):
    tau: float = 1.0
    lambda_: float = 10.0    # choose from [10.0, 100.0, 1000.0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
Once the perturbation intensity is decided, we can run the following command:
 - `python 3_main.py --connected={}`

## Explanation Evaluation
The folder `xai_evaluation` contains the scripts to train, evaluate and plot the explanations obtained with our XAI method on a 3-layer GIN architecture.

### Datasets
The folder `datasets` contains the raw data used for our experiments: `ba_2motifs` and `Mutagenicity (MUTAG_0)`. The first one can be directly obtained using commands from common [libraries](https://diveintographs.readthedocs.io/en/latest/xgraph/dataset.html#dig.xgraph.dataset.SynGraphDataset). The latter was manually downloaded from [this](https://github.com/chrisjtan/gnn_cff/tree/main/datasets/Mutagenicity_0) repository.
Before training L2XGNN, we need to preprocess the `MUTAG_0` dataset using the following command:
 - `python 0_preprocess_Mutagenicity_data`

### Train L2XGNN
To train L2XGNN on a 3-layer GIN architecture (as in the paper), use the following command:
 - `python 1_l2xgnn_train.py --dataset={} --model={} --connected={} --ratio={}`
	 - `dataset`: choose between `ba_2motifs` and `Mutagenicity`.
	 - `model`: this parameter can be used to choose whether we want to explain a GIN or a GCN architecture. You can choose between `L2XGIN` and `L2XGCN` respectively (default `L2XGIN`).
	 - `connected`: parameter to decide between connected and disconnected subgraphs (default value `True`).
	 - `ratio`: ratio of restrained edges (float between 0.1 and 0.9).
### Evaluate L2XGNN
Once the explanations are obtained, we can compute the explanation accuracy in comparison with the available ground-truth motifs:
 - `python 2_l2xgnn_evaluate.py --dataset={} --model={} --connected={} --ratio={}`
### Plot explanations
Finally, by running the next command, we can generate and save the images of the explanations learned by our method.
 - `python 3_plot_explanations.py --dataset={} --model={} --connected={} --ratio={}`
 
 
## Reference

```bibtex

@article{serra2022l2xgnn,
  title={Learning to Explain Graph Neural Networks},
  author={Serra, Giuseppe and Niepert, Mathias},
  journal={arXiv preprint arXiv:2209.14402},
  year={2022}
}

```


