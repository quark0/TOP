# TOP
Transductive Learning over the Product Graph

## Description
This piece of code implements the content-aware link prediction algorithm described in

> Liu, Hanxiao, and Yiming Yang. "Bipartite Edge Prediction via Transductive Learning over Product Graphs." Proceedings of the 32nd International Conference on Machine Learning (ICML). 2015.

The program is going to carry out transductive learning over the Cartesian product graph with the heat diffusion kernel, which gives the best average empirical performace. 

## Usage
The program takes four files as its input:

A sparse graph G on the left and a sparse graph H on the right in the following format
```
vertexIn_G anotherVertexIn_G edgeStrength
```
Cross-graph links for training and testing
```
vertexIn_G vertexIn_H linkStrength
```
For example,
G could be the social network among the users
and H could be the graph of movie-movie similarity induced from the movie genres.
In this case, cross-graph links may correspond to user-movie ratings.
  
By default, the program reads configurations specified in `cfg.ini`. The configuration file should be self-explanatory. Here is a sample pipeline for execution and evaluation:
```
make && ./train && python eval.py data/cmu/link.test.txt link.predict.txt
```

## Author
Hanxiao Liu, Carnegie Mellon University
