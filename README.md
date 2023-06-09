# Foundation Model for Endoscopy Video Analysis
<!-- select Model and/or Data and/or Code as needed>
### Welcome to OpenMEDLab! 👋

<!--
**Here are some ideas to get you started:**
🙋‍♀️ A short introduction - what is your organization all about?
🌈 Contribution guidelines - how can the community get involved?
👩‍💻 Useful resources - where can the community find your docs? Is there anything else the community should know?
🍿 Fun facts - what does your team eat for breakfast?
🧙 Remember, you can do mighty things with the power of [Markdown](https://docs.github.com/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
-->


<!-- Insert the project banner here -->

[//]: # (<div align="center">)

[//]: # (    <a href="https://"><img width="1000px" height="auto" src="https://github.com/openmedlab/sampleProject/blob/main/banner_sample.png"></a>)

[//]: # (</div>)

[//]: # (---)

<!-- Select some of the point info, feel free to delete -->

[//]: # ([![Twitter]&#40;https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fopendilab&#41;]&#40;https://twitter.com/opendilab&#41;)

[//]: # ([![PyPI]&#40;https://img.shields.io/pypi/v/DI-engine&#41;]&#40;https://pypi.org/project/DI-engine/&#41;)

[//]: # (![Conda]&#40;https://anaconda.org/opendilab/di-engine/badges/version.svg&#41;)

[//]: # (![Conda update]&#40;https://anaconda.org/opendilab/di-engine/badges/latest_release_date.svg&#41;)

[//]: # (![PyPI - Python Version]&#40;https://img.shields.io/pypi/pyversions/DI-engine&#41;)

[//]: # (![PyTorch Version]&#40;https://img.shields.io/badge/dynamic/json?color=blue&label=pytorch&query=%24.pytorchVersion&url=https%3A%2F%2Fgist.githubusercontent.com/PaParaZz1/54c5c44eeb94734e276b2ed5770eba8d/raw/85b94a54933a9369f8843cc2cea3546152a75661/badges.json&#41;)

[//]: # ()
[//]: # ()
[//]: # (![Loc]&#40;https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/loc.json&#41;)

[//]: # (![Comments]&#40;https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/3690cccd811e4c5f771075c2f785c7bb/raw/comments.json&#41;)

[//]: # ()
[//]: # (![Style]&#40;https://github.com/opendilab/DI-engine/actions/workflows/style.yml/badge.svg&#41;)

[//]: # (![Docs]&#40;https://github.com/opendilab/DI-engine/actions/workflows/doc.yml/badge.svg&#41;)

[//]: # (![Unittest]&#40;https://github.com/opendilab/DI-engine/actions/workflows/unit_test.yml/badge.svg&#41;)

[//]: # (![Algotest]&#40;https://github.com/opendilab/DI-engine/actions/workflows/algo_test.yml/badge.svg&#41;)

[//]: # (![deploy]&#40;https://github.com/opendilab/DI-engine/actions/workflows/deploy.yml/badge.svg&#41;)

[//]: # ([![codecov]&#40;https://codecov.io/gh/opendilab/DI-engine/branch/main/graph/badge.svg?token=B0Q15JI301&#41;]&#40;https://codecov.io/gh/opendilab/DI-engine&#41;)

[//]: # ()
[//]: # (![GitHub Org's stars]&#40;https://img.shields.io/github/stars/opendilab&#41;)

[//]: # ([![GitHub stars]&#40;https://img.shields.io/github/stars/opendilab/DI-engine&#41;]&#40;https://github.com/Med-AIR/Endo-FM/stargazers&#41;)

[//]: # ([![GitHub forks]&#40;https://img.shields.io/github/forks/opendilab/DI-engine&#41;]&#40;https://github.com/Med-AIR/Endo-FM/network&#41;)

[//]: # (![GitHub commit activity]&#40;https://img.shields.io/github/commit-activity/m/opendilab/DI-engine&#41;)

[//]: # ([![GitHub issues]&#40;https://img.shields.io/github/issues/opendilab/DI-engine&#41;]&#40;https://github.com/opendilab/DI-engine/issues&#41;)

[//]: # ([![GitHub pulls]&#40;https://img.shields.io/github/issues-pr/opendilab/DI-engine&#41;]&#40;https://github.com/opendilab/DI-engine/pulls&#41;)

[//]: # ([![Contributors]&#40;https://img.shields.io/github/contributors/opendilab/DI-engine&#41;]&#40;https://github.com/opendilab/DI-engine/graphs/contributors&#41;)

[//]: # ([![GitHub license]&#40;https://img.shields.io/github/license/opendilab/DI-engine&#41;]&#40;https://github.com/Med-AIR/Endo-FM/blob/master/LICENSE&#41;)

[//]: # (Updated on 2023.06.09)



This repository provides the official PyTorch implementation of the paper [**Foundation Model for Endoscopy Video Analysis via Large-scale Self-supervised Pre-train**](TBA)
by [Zhao Wang](https://kyfafyd.wang)\*, Chang Liu\*, [Shaoting Zhang](http://www.qingyuan.sjtu.edu.cn/a/Shaoting-Zhang.html)†, and [Qi Dou](http://www.cse.cuhk.edu.hk/~qdou)†.

<div align="center">
    <a href="https://"><img width="800px" height="auto" src="assets/framework.png"></a>
</div>

## Key Features


[//]: # (key feature bulletin points here)
- First foundation model for endoscopy video analysis.
- A large-scale endoscopic video dataset with over 32k video clips.
- Support 3 types of downstream tasks, including classification, segmentation, and detection.

## Links

- [Paper](TBA)
- [Model](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/EZh5mWE5CL1BpaJ1bXuokfYBDM2VaMknqG7YpaQBRgAvdQ?e=e2rVYW)
- [Code](https://github.com/med-air/Endo-FM) 
<!-- [Code] may link to your project at your institute>


<!-- give a introduction of your project -->

## Details

> Recent foundation models have exhibited remarkable success in various downstream tasks, such as disease diagnosis and report generation. However, a foundation model for endoscopic videos is lacking. In this paper, we propose Endo-FM, a foundation model specifically designed for endoscopic video analysis. First, we build a video transformer as Endo-FM, which captures both local and global long-range dependencies across spatial and temporal dimensions. Second, we pre-train our Endo-FM using global and local views to be robust to spatial-temporal changes and discriminative across different videos. To achieve this, we construct a large-scale endoscopy video dataset by combining all publicly available datasets and a new private one. This dataset consists of over 32K video clips (5M frames), encompassing varying modalities, target organs, and disease types. Our pre-trained Endo-FM achieves promising performance on downstream tasks, surpassing state-of-the-art methods by a significant margin.

<!-- Insert a pipeline of your algorithm here if got one -->


[//]: # (More intro text here.)


## Datasets

<div align="center">
    <a href="https://"><img width="800px" height="auto" src="assets/dataset.png"></a>
</div>

We utilize 6 public and 1 private datasets for pre-training and 3 datasets as the downstream tasks.
Except for SUN-SEG, we provide our preprocessed data for pre-training and downstream tasks, you can directly download via the following links:
- [pre-training](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/EThG3T11jIxJg4eDb-Ku9xEB6LjZBcWHrseGrNu4PK2orQ?e=zWJPxR)
- [downstream](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/EcubwoZxij1AhG6VM3G9bT8BFplXFN2tp9yWj5HfLm3vug?e=DqyXq1)

For SUN-SEG, you need first request the original videos following [this instruction](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md).
Then, you can transfer SUN-SEG for pre-training videos by the following:
```bash
cd Endo-FM/data
python sun.py
python sun_seg.py
python trans_videos_pretrain.py
```
Finally, generating the video list `pretrain/train.csv` for pre-training by the following:
```bash
cd Endo-FM/data
python gencsv.py
```

#### Pre-training
- [Colonoscopic](http://www.depeca.uah.es/colonoscopy_dataset/)
- [SUN-SEG](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md)
- [LPPolypVideo](https://github.com/dashishi/LDPolypVideo-Benchmark)
- [Hyper-Kvasir](https://datasets.simula.no/hyper-kvasir/)
- [Kvasir-Capsule](https://datasets.simula.no/kvasir-capsule/)
- [CholecTriplet](https://cholectriplet2021.grand-challenge.org/)
- [Ours Private](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/EThG3T11jIxJg4eDb-Ku9xEB6LjZBcWHrseGrNu4PK2orQ?e=zWJPxR)

#### Downstream
- [PolypDiag](https://github.com/tianyu0207/weakly-polyp)
- [CVC-12k](https://polyp.grand-challenge.org/Databases/)
- [KUMC](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/FCBUOR)


## Get Started

#### Main Requirements
- torch==1.8.0
- torchvision==0.9.0
- pillow==6.2.2
- timm==0.4.12

#### Installation
We suggest using Anaconda to setup environment on Linux, if you have installed anaconda, you can skip this step.

```shell
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh && zsh Anaconda3-2020.11-Linux-x86_64.sh
```

Then, we can install packages using provided `environment.yaml`.

```shell
cd Endo-FM
conda env create -f environment.yaml
conda activate endofm
```

#### Pre-trained Weights
You can directly download our pre-trained Endo-FM via this [link](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155167044_link_cuhk_edu_hk/EZh5mWE5CL1BpaJ1bXuokfYBDM2VaMknqG7YpaQBRgAvdQ?e=e2rVYW) and put it under `checkpoints/` for downstream fine-tuning.
Also, we provide the pre-trained weights of 3 downstream tasks via this [link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155167044_link_cuhk_edu_hk/Ei7k6w3Qic9PmSpuoRLwPbsBw1bLeh-3DrIuSAj4yedabA?e=uiDpCy) for direct downstream testing.

[//]: # (#### Preprocess)


#### Pre-training
```shell
cd Endo-FM
wget -P checkpoints/ https://github.com/kahnchana/svt/releases/download/v1.0/kinetics400_vitb_ssl.pth
bash scripts/train_clips32k.sh
```

#### Downstream Fine-tuning
```shell
# PolypDiag (Classification)
cd Endo-FM
bash scripts/eval_finetune_polypdiag.sh

# CVC (Segmentation)
cd Endo-FM/TransUNet
python train.py

# KUMC (Detection)
cd Endo-FM/STMT
python setup.py build develop
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    tools/train_net.py \
    --master_port=$((RANDOM + 10000)) \
    --config-file configs/STFT/kumc_R_50_STFT.yaml \
    OUTPUT_DIR log_dir/kumc_finetune
```

#### Direct Downstream Testing
```shell
# PolypDiag (Classification)
cd Endo-FM
bash scripts/test_finetune_polypdiag.sh

# CVC (Segmentation)
cd Endo-FM/TransUNet
python train.py --test

# KUMC (Detection)
cd Endo-FM/STMT
python setup.py build develop
python -m torch.distributed.launch \
    --nproc_per_node=1 \
    tools/test_net.py \
    --master_port=$((RANDOM + 10000)) \
    --config-file configs/STFT/kumc_R_50_STFT.yaml \
    MODEL.WEIGHT kumc.pth \
    OUTPUT_DIR log_dir/kumc_finetune
```

## 🙋‍♀️ Feedback and Contact

For further questions, pls feel free to contact [Zhao Wang](mailto:zwang21@cse.cuhk.edu.hk).


## 🛡️ License

This project is under the Apache License 2.0 license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgement

Our code is based on [DINO](https://github.com/facebookresearch/dino), [TimeSformer](https://github.com/facebookresearch/TimeSformer), [SVT](https://github.com/kahnchana/svt), [TransUNet](https://github.com/Beckschen/TransUNet), and [STFT](https://github.com/lingyunwu14/STFT). Thanks them for releasing their codes.

## 📝 Citation

If you find this code useful, please cite in your research papers.
```
@inproceedings{
    wang2023foundation,
    title={Foundation Model for Endoscopy Video Analysis via Large-scale Self-supervised Pre-train},
    author={Zhao Wang and Chang Liu and Shaoting Zhang and Qi Dou},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    pages={},
    year={2023},
    organization={Springer}
}
```
