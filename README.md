# KeHGN-R

KeHGN-R is the official github repository for paper :

### Enviroment required

This repository can be run under:

> python=3.9.12, pytorch=1.11.0+cu113, sklearn, scipy, pandas

### Quick start

##### Dataset

##### Training

A example to train the KeGCN_R：

```
python main.py --model KeHGNN --dataset full -d 1000 -lr 0.0001 --device 0
```

A example to train the KeGCN:

```
python main.py --model KeHGN --dataset full -d 1000 -lr 0.0001 --device 0
```

A example to train the KeGCN_R w/o ke：

```
python main.py --model KeHGNN -wke --dataset full -d 1000 -lr 0.0001 --device 0
```
