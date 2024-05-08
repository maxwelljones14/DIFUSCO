# Approximating Graph Algorithms with Graph Neural Networks


Note: This repo was adapted from DIFUSCO: Graph-based Diffusion Solvers for Combinatorial Optimization.
See ["DIFUSCO for MST/SSP/MinCut: Graph-based Diffusion Solvers for Combinatorial Optimization based on DIFUSCO"].

![440703644_1480855556189748_1409073861208133370_n](https://github.com/maxwelljones14/DIFUSCO/assets/56135118/5ba23a0b-a2d1-44bf-b1ea-d515623d8f50)


## Setup

```bash
conda env create -f environment.yml
conda activate difusco
```

Running TSP experiments requires installing the additional cython package for merging the diffusion heatmap results:

```bash
cd difusco/utils/cython_merge
python setup.py build_ext --inplace
cd -
```

## Codebase Structure

* `difusco/pl_meta_model.py`: the code for a meta pytorch-lightning model for training and evaluation.
* `difusco/pl_tsp_model.py`: the code for the TSP problem
* `difusco/pl_mis_model.py`: the code for the MIS problem
* `difusco/trian.py`: the handler for training and evaluation

## Data

Please check the `data` folder.

## Reproduction
Our reproducing scripts are the same as the ones in the main file - the only difference is that in the 'train.py' call, you can add arguments --mst_only, or --mincut_only, or --dijkstra_only. We also provide scripts for creating the data for these graph problems in the data file
Please check the [reproducing_scripts](reproducing_scripts.md) for more details.

## Pretrained Checkpoints

Please download the pretrained model checkpoints from [here](https://drive.google.com/drive/folders/1IjaWtkqTAs7lwtFZ24lTRspE0h1N6sBH?usp=sharing).

## Reference

If you found this codebase useful, please consider citing the paper:

```
@inproceedings{
    sun2023difusco,
    title={{DIFUSCO}: Graph-based Diffusion Solvers for Combinatorial Optimization},
    author={Zhiqing Sun and Yiming Yang},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=JV8Ff0lgVV}
}
```
