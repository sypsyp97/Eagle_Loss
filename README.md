# Eagle_Loss
[![arXiv](https://img.shields.io/badge/arXiv-2403.10695-b31b1b.svg)](http://arxiv.org/abs/2403.10695)
[![SPIE Journal](https://img.shields.io/badge/SPIE%20Journal-10.1117%2F1.JMI.12.1.014001-blue.svg)](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-12/issue-1/014001/EAGLE--an-edge-aware-gradient-localization-enhanced-loss-for/10.1117/1.JMI.12.1.014001.full)

PyTorch implementation of the paper **"EAGLE: An Edge-Aware Gradient Localization Enhanced Loss for CT Image Reconstruction"**, officially published in [SPIE Journal of Medical Imaging](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-12/issue-1/014001/EAGLE--an-edge-aware-gradient-localization-enhanced-loss-for/10.1117/1.JMI.12.1.014001.full). This repository includes the code for our novel Eagle-Loss function, designed to improve the sharpness of reconstructed CT images.

## Installation

To ensure compatibility, please install the necessary packages using the following commands to create a conda environment and install eagle_loss package.:

```bash
git clone https://github.com/sypsyp97/Eagle_Loss.git
conda env create -f environment.yml
conda activate eagle_loss
cd Eagle_Loss
pip install -e .
```

## Data
FOV extension data can be downloaded [here](https://drive.google.com/file/d/11Pkdw420Al4ubLKce4fNRrEqAD_37Gfg/view?usp=sharing).

## Usage
You can find the example usage in [`example.py`](examples/example.py).

## Citation
Please cite the following paper and star this project if you use this repository in your research. Thank you!
```
@article{sun2025eagle,
  title={EAGLE: an edge-aware gradient localization enhanced loss for CT image reconstruction},
  author={Sun, Yipeng and Huang, Yixing and Yang, Zeyu and Schneider, Linda-Sophie and Thies, Mareike and Gu, Mingxuan and Mei, Siyuan and Bayer, Siming and Z{\"o}llner, Frank G and Maier, Andreas},
  journal={Journal of Medical Imaging},
  volume={12},
  number={1},
  pages={014001--014001},
  year={2025},
  publisher={Society of Photo-Optical Instrumentation Engineers}
}
```
