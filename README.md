# ICL
This is the code for ICL.

## Requirements
* Python 3.6
* numpy==1.19.2
* sklearn==0.23.2
* torch==1.6.0
* torch-geometric==1.6.3
* torchmeta==1.7.0
* torchvision==0.7.0

## Data
The data should be placed in the path `./data/` which should be the same with the code in `dataset.py`. For example, `./data/cv/ImageNet-2/`.

## ICL
We provide four variants of ICL:
* **ICL:** `train_ICL_MAML_RL.py`
* **ICL (w/o LRL):** `train_ICL_MAML.py`
* **ICL (w/o M):** `train_ICL_RL.py`
* **ICL (w/o M+LRL):** `train_ICL.py`

## Usage
Commands for training and testing the method on CV and GRL datasets.

Train a resnet18 on `ImageNet-2`:
```
python -u main.py -m cv -d ImageNet-2 -en resnet18 -a 0.5
```
Train a gcn on `PROTEINS_full`:
```
python -u main.py -m graph -d PROTEINS_full -en gcn -a 0.5
```
More detailed parameter information please refer to `main.py`.  
We provide some key parameter description:
|Name|Description|
|-|-|
|mode|The mode of ICL: cv or graph (defaults to `cv`).
|dataset|The dataset name (defaults to `ImageNet-2`).|
|encoder|The encoder: resnet18 or gcn (defaults to `resnet18`)|
|alpha|The growth rate of the data $\alpha$ (defaults to `0.5`).|

## Output
We train the different variants on the same encoder trained on the old data, and report the `training time`, `converge epoch`, and the `accuracy`.
For instance:
```
Begin!
...
Inference!
Epoch 0    , Loss 2.4097, time 0.8747, total time 0.8747
Epoch 10   , Loss 2.3572, time 0.6263, total time 7.0705
...
Eval!
Final Epoch 31   , Loss 1.7602, time 0.6178, total time 19.9385, acc 0.7187, acc-old 0.6961, acc-new 0.7108
ICL-MAML-RL!
Epoch 0    , Loss 2.9381, time 1.8081, total time 1.8081
Epoch 10   , Loss 2.8510, time 1.7075, total time 16.2273
...
Done!
```
