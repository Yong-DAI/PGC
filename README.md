

This repository contains the implementation details of our Prompt Generation and Catalyzing approach for continual learning with transformer backbone.

“Dual-Attention Based Prompt Generation and Catalyzing for Instance-wise Continual Learning”

## Requirements
The code is written for python `3.8.18`, but should work for other version with some modifications.
```
pip install -r requirements.txt
```
## Data preparation
If you already have CIFAR-100/ImageNet-R/CUB-200, pass your dataset path to the `--data-path` argument during execution
(If they aren't ready they will automatically get downloaded to the data-path mentioned when the `download` argument is kept True in `datasets.py`).



## Training
Use the following command for training:

```
export TOKENIZERS_PARALLELISM=false
python -m main <cifar100_pgt > \
        --num_tasks 10 \
        --data-path /local_datasets/ \
        --output_dir ./output 
```


## Acknowledgement

This repo is heavily based on the  [DualPrompt](https://github.com/JH-LEE-KR/dualprompt-pytorch) and [ConvP](https://cvir.github.io/projects/convprompt) Implementations. We thank them for their wonderful work!






