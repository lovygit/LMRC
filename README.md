# Experimental Source Code of LMRC


## Data root

- ./data/: save dataset
- ./model/: save trained models

## Source code

### Base network

- ./models/: source code of base network architecture, including ResNet and Simple-CNN

### Helper function

- utils.py: data loader, data transformer and learing rate scheduler
- load\_data_online.py: overwrite the load data function of PyTorch for online learning
- label\_mapping.py: implementation of Label Mapping algorithm

### model training

- EWC.py and EWC\_data_pool.py 
- LMRC.py and LMRC\_data_pool.py
- LWF.MC.py and LWF.MC\_data_pool.py
- fine\_tune.py and fine\_tune\_data_pool.py
- LM.py
- LWF.MT.py
