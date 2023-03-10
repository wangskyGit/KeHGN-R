# KeGCN-R

KeGCN-R is the official github repository for paper :

### Enviroment required

This repository can be run under:

> python=3.9.12, pytorch=1.11.0+cu113, sklearn, scipy, pandas

### Quick start

##### Dataset

'main" stands for the MBM dataset, 'mini' stands for the SME dataset, 'entre' stands for the GEM dataset. All the datasets are collected from [CSMAR](https://cn.gtadata.com/), and **please use the data according to the related regulations of the CSMAR database**. Datsets can be download from : [GoogleDrive-datsets](https://drive.google.com/file/d/1E1IpDYBKLiJH1bERtjm4E8fdfRTXbJQR/view?usp=sharing), please download and unzip it to the current folder before running the code. 

The datasets including adjacent matrix (*.npy) of all company subgraphs and csv files (fv.csv) with all features and labels. Pay attention that there is also a feature named 'year' in the fv.csv which is used to construct the trust-worthy test datasets and should not be used as the node attributes.

**More details and describtion of the datasets will be pubnished when the paper gets acceptted.**

##### FKG knowledge embedding pretraining

We use [DGL-KE](https://dglke.dgl.ai/doc/) for knowledge embedding learning, and the pretraining result can be found in [GoogleDrive-KE](https://drive.google.com/file/d/1pvCNNMzA8_NeJViXc-KvMZXItsHR91pO/view?usp=sharing), please download and unzip it to the current folder before running the code.
##### Training

A example to train the KeGCN_R：

```
python main.py --model KeGCNR --dataset main -d 1000 -lr 0.0001 --device 0
```

A example to train the KeGCN:

```
python main.py --model KeGCN --dataset main -d 1000 -lr 0.0001 --device 0
```

A example to train the KeGCN_R w/o ke：

```
python main.py --model KeGCNR -wke --dataset main -d 1000 -lr 0.0001 --device 0
```
