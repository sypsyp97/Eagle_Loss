# Eagle_Loss
[![arXiv](https://img.shields.io/badge/arXiv-2403.10695-b31b1b.svg)](http://arxiv.org/abs/2403.10695)


Official implementation of the paper "EAGLE: An Edge-Aware Gradient Localization Enhanced Loss for CT Image Reconstruction". This repository includes the code for our novel Eagle-Loss function, designed to improve the sharpness of reconstructed CT image.

## Requirements

The Eagle_Loss code is developed using Python 3.11 and PyTorch 2.0.0. To ensure compatibility, please install the necessary packages as listed in the "environment.yml" file. Use the following commands to create and activate a conda environment:

```bash
conda env create -f environment.yml
conda activate eagle_loss
```


## Data
FOV extension data can be downloaded from the following link:
https://drive.google.com/file/d/11Pkdw420Al4ubLKce4fNRrEqAD_37Gfg/view?usp=sharing

## Code Structure

This repository is organized as follows:

- `dataset.py`: This script is responsible for handling the dataset.

- `eagle_loss.py`: Contains the implementation of the Eagle-Loss function.

- `model.py`: Defines the architecture of the U-Net that is used for FOV extension.

## To-Do List

- [ ] Training script.
- [ ] Pre-trained model weights.
- [ ] Usage examples.

## Citation

```
@misc{sun2024eagle,
      title={EAGLE: An Edge-Aware Gradient Localization Enhanced Loss for CT Image Reconstruction}, 
      author={Yipeng Sun and Yixing Huang and Linda-Sophie Schneider and Mareike Thies and Mingxuan Gu and Siyuan Mei and Siming Bayer and Andreas Maier},
      year={2024},
      eprint={2403.10695},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
