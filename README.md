# AadaptiveS-tgn

(sigir2021)

## Runing the experiments

### dataset and preprocessing

#### Data downloading

* [Reddit](http://snap.stanford.edu/jodie/reddit.csv)
* [Wikipedia](http://snap.stanford.edu/jodie/wikipedia.csv)

#### Data preprocessing

We use the dense `npy` format to save the features in binary format. If edge features or nodes features are absent, it will be replaced by a vector of zeros. 

* Step1 :

```{bash}
python process.py  --data wikipedia/reddit
```

* Step2:

```python
python BuildDglGraph.py --data wikipedia/reddit
```



#### Use your own data

Put your data under `processed` folder. The required input data includes `ml_${DATA_NAME}.csv`, `ml_${DATA_NAME}.npy` and `ml_${DATA_NAME}_node.npy`. They store the edge linkages, edge features and node features respectively. 

The `CSV` file has following columns

```
u, i, ts, label, idx
```

, which represents source node index, target node index, time stamp, edge label and the edge index. 

`ml_${DATA_NAME}.npy` has shape of [#temporal edges + 1, edge features dimention]. Similarly, `ml_${DATA_NAME}_node.npy` has shape of [#nodes + 1, node features dimension].


All node index starts from `1`. The zero index is reserved for `null` during padding operations. So the maximum of node index equals to the total number of nodes. Similarly, maxinum of edge index equals to the total number of temporal edges. The padding embeddings or the null embeddings is a vector of zeros.

### Requirements

* python>=3.7

* Dependency

  ```python
  dgl==0.5.2
  numpy==1.18.1
  scikit-learn==0.23.2
  torch==1.5.0
  tqdm==4.48.2
  ```

### Training

```python
python train.py --lr=1e-4 --epoch=30
```

#### General flag

Optional arguments are described in args.py.



