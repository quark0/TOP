# TOP
Transductive Learning over the Product Graph

## Description
This piece of code implements the content-aware link prediction algorithm described in

> Liu, Hanxiao, and Yiming Yang. "Bipartite Edge Prediction via Transductive Learning over Product Graphs." Proceedings of the 32nd International Conference on Machine Learning (ICML-15). 2015.

The program is going to carry out transductive learning over the Cartesian product graph with the heat diffusion kernel, which gives the best average empirical performace. 

## Usage
By default, the program reads the configurations in `cfg.ini`. The configuration file should be self-explanatory. Here is a sample pipeline for execution and evaluation:
```
make && ./train && python eval.py data/cmu/link.test.txt link.predict.txt
```

## Author
Hanxiao Liu, Carnegie Mellon University
