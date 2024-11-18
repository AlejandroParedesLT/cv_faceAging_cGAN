# Conditional generative adversarial network
### Alejandro Paredes La Torre

The following implementation was based in:
Author: Yuchao Gu
[《Face Aging with Identity-Preserved Conditional Generative Adversarial Networks》](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Face_Aging_With_CVPR_2018_paper.pdf)


---

## Usage

### Data Preparation

1. Download the [wiki dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) and unzip.
2. Use preprocess/preprocess_cacd.py to crop and align face region.
3. Use preprocess/gentraingroup.py to generate training list, please refer to data/wiki-lists as an example.

### Training

1. Pretrain Age Classify model

``` python
python pretrain_alexnet.py
```

2. Train IPCGANs

``` python
python IPCGANS_train.py
```

### Visualize

![](./readmeDisplay/1.png)

### Test the app
``` python
python app.py 
```

### Dependencies

This code depends on the following libraries:

* Python 3.6
* Pytorch 0.4.1
* PIL
* TensorboardX
* Flask

### Structure
```
IPCGANs
│
├── checkpoint  # path to store logging and parameters
│ 
├── data # path to specify training data groups and testing data groups
│ 
├── data_generator # pytorch dataloader for training
│ 
├── model # path to define model
│ 
├── utils
│
├── preprocess
│  
├── app.py # server scripts
│
├── demo.py # api for server 
│  
├── pretrain_alexnet.py
│
├── IPCGANS_train.py 
│
└── README.md # introduce to this project
```



