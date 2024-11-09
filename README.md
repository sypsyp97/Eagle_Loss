# Eagle_Loss
[![arXiv](https://img.shields.io/badge/arXiv-2403.10695-b31b1b.svg)](http://arxiv.org/abs/2403.10695)

PyTorch implementation of the paper ["EAGLE: An Edge-Aware Gradient Localization Enhanced Loss for CT Image Reconstruction"](https://arxiv.org/abs/2403.10695). This repository includes the code for our novel Eagle-Loss function, designed to improve the sharpness of reconstructed CT image.

## Installation

To ensure compatibility, please install the necessary packages using the following commands to create a conda environment and install eagle_loss package.:

```bash
conda env create -f environment.yml
conda activate eagle_loss
cd eagle_loss
pip install -e .
```

## Data
FOV extension data can be downloaded [here](https://drive.google.com/file/d/11Pkdw420Al4ubLKce4fNRrEqAD_37Gfg/view?usp=sharing).

## Usage
You can find the example usage in [`example.py`](examples/example.py).

## Citation

```
@article{sun2024eagle,
  title={EAGLE: An Edge-Aware Gradient Localization Enhanced Loss for CT Image Reconstruction},
  author={Sun, Yipeng and Huang, Yixing and Schneider, Linda-Sophie and Thies, Mareike and Gu, Mingxuan and Mei, Siyuan and Bayer, Siming and Maier, Andreas},
  journal={arXiv preprint arXiv:2403.10695},
  year={2024}
}
```
