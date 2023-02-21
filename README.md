# HoscPool
Higher-order clustering and pooling for GNNs

Paper accepted at the 31st ACM International Conference on Information and Knowledge Management (CIKM, 2022). 
https://arxiv.org/abs/2209.03473

In this repo, we provide: 
(1) the code for the pooling operator itself in [hoscpool.py](/hoscpool.py). It can be used similarly to MinCutPool or DiffPool and will soon be available in Pytorch Geometric. 
(2) a short [example](/example.py) of how to use hoscpool for an arbitrary graph classification task.  


### Set up 
If needed, install the required packages: torch and torch-geometric. 

### Citation 
Please cite the original paper if you are using HoscPool in your work. 
```
@inproceedings{duval2022higher,
  title={Higher-order clustering and pooling for graph neural networks},
  author={Duval, Alexandre and Malliaros, Fragkiskos},
  booktitle={Proceedings of the 31st ACM International Conference on Information \& Knowledge Management},
  pages={426--435},
  year={2022}
}
```
