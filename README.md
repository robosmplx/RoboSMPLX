<div align="center">

# Towards Robust and Expressive Whole-body Human Pose and Shape Estimation

<div>
    <a href='' target='_blank'>Hui En Pang</a>&emsp;
    <a href='https://caizhongang.github.io/' target='_blank'>Zhongang Cai</a>&emsp;
    <a href='https://yanglei.me/' target='_blank'>Lei Yang</a>&emsp;
    <a href='' target='_blank'>Qingyi Tao</a>&emsp;
    <a href='' target='_blank'>Zhonghua Wu</a>&emsp;
    <a href='https://scholar.google.com/citations?user=9vpiYDIAAAAJ&hl=en' target='_blank'>Tianwei Zhang</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>
</div>
<div>
    S-Lab, Nanyang Technological University
</div>

<strong><a href='https://nips.cc/Conferences/2023' target='_blank'>NeurIPS 2023</a></strong>

<h4 align="center">
  <a href="" target='_blank'>[arXiv]</a> •
  <a href="" target='_blank'>[Slides]</a>
</h4>

## Getting started
### [Installation](#installation) | [Train](#train) | [Evaluation](#evaluation)


</div>

## Introduction

This repo is official PyTorch implementation of Towards Robust and Expressive Whole-body Human Pose and Shape Estimation (NeurIPS2023). 


<p align="center">
    <!-- <img src="resources/dance3.gif" width="99%"> -->
    <img src="assets/teaser.png" width="99%">
    <!-- <img src="resources/dance001.gif" width="80%"> -->
</p>


## Installation

```bash
# 1. Create a conda virtual environment.
conda create -n robosmplx python=3.8 -y
conda activate robosmplx

# 2. Install dependenciesv
conda install ffmpeg
conda install pytorch=1.8.0 torchvision cudatoolkit=10.2 -c pytorch
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d

pip install mmcv_full-1.5.3-cp38-cp38-manylinux1_x86_64.whl
rm mmcv_full-1.7.1-cp38-cp38-manylinux1_x86_64.whl

# 3. Pull our code
git clone https://github.com/robosmplx/RoboSMPLX
cd RoboSMPLX
pip install -v -e .
```



## Train (TBA)

### Training with a single / multiple GPUs

```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --no-validate
```
Example: using 1 GPU to train RoboSMPLX.
```shell
python tools/train.py ${CONFIG_FILE} ${WORK_DIR} --gpus 1 --no-validate
```

### Training with Slurm

If you can run RoboSMPLX on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_train.sh`.

```shell
./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM} --no-validate
```

Common optional arguments include:
- `--resume-from ${CHECKPOINT_FILE}`: Resume from a previous checkpoint file.
- `--no-validate`: Whether not to evaluate the checkpoint during training.

Example: using 8 GPUs to train RoboSMPLX on a slurm cluster.
```shell
./tools/slurm_train.sh my_partition my_job configs/robosmplx/resnet50_hmr_pw3d.py work_dirs/hmr 8 --no-validate
```

You can check [slurm_train.sh](https://github.com/open-mmlab/mmhuman3d/tree/main/tools/slurm_train.sh) for full arguments and environment variables.


### Training configs
```bash
sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR} ${GPU_NUM}

# Stage 1a: train hand
sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} configs/robosmplx/robosmplx_hand.py expts/robosmplx_hand 8

# Stage 1b: train face
sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} configs/robosmplx/robosmplx_face.py expts/robosmplx_face 8

# Stage 1c: train body
sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} configs/robosmplx/robosmplx_body.py expts/robosmplx_body 8

# Stage 2: train wholebody
sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} configs/robosmplx/robosmplx.py expts/robosmplx_wholebody 8

# Stage 2b: FT wholebody on AGORA
sh tools/slurm_train.sh ${PARTITION} ${JOB_NAME} configs/robosmplx/robosmplx_agora.py expts/robosmplx_wholebody_agora 8
```


## Evaluation (TBA)

For our robustness study, we evaluated on 10 different augmentations:
- Vertical translation
- Horizontal translation
- Scale
- Low Resolution
- Rotation
- Hue
- Sharpness
- Grayness
- Contrast
- Brightness

### Evaluate with a single GPU / multiple GPUs

```shell
python tools/test.py ${CONFIG} --work-dir=${WORK_DIR} ${CHECKPOINT} --metrics=${METRICS} --augmentation=${AUGMENTATION}
```
Example:
```bash
python tools/test.py configs/robosmplx/robosmplx.py --work-dir=work_dirs/robosmplx work_dirs/robosmplx/latest.pth 

# Evaluation under augmentation
python tools/test.py configs/robosmplx/robosmplx.py --work-dir=work_dirs/robosmplx work_dirs/robosmplx/latest.pth --augmentation=rotation

```

### Evaluate with slurm

If you can run MMHuman3D on a cluster managed with [slurm](https://slurm.schedmd.com/), you can use the script `slurm_test.sh`.

```shell
./tools/slurm_test.sh ${PARTITION} ${JOB_NAME} ${CONFIG} ${WORK_DIR} ${CHECKPOINT} --metrics ${METRICS}
```
Example:
```bash
./tools/slurm_test.sh my_partition test_robosmplx configs/robosmplx/robosmplx.py work_dirs/robosmplx work_dirs/robosmplx/latest.pth 1 

# Evaluation under augmentation
./tools/slurm_test.sh my_partition test_robosmplx configs/robosmplx/robosmplx.py work_dirs/robosmplx work_dirs/robosmplx/latest.pth 1 --augmentation rotation
```


## Citation
If you find our work useful for your research, please consider citing the paper:
```
@inproceedings{
  title={Towards Robust and Expressive Whole-body Human Pose and Shape Estimation},
  author={Pang, Hui En and Cai, Zhongang and Yang, Lei and Qingyi, Tao and Zhonghua, Wu and Zhang, Tianwei and Liu, Ziwei},
  booktitle={NeurIPS},
  year={2023}
}
```

## License

Distributed under the S-Lab License. See `LICENSE` for more information.

## Acknowledgements

This research/project is supported by the National Research Foundation, Singapore under its AI Singapore Programme. This study is also supported by the Ministry of Education, Singapore, under its MOE AcRF Tier 2 (MOE-T2EP20221-0012), NTU NAP, and under the RIE2020 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) Funding Initiative, as well as cash and in-kind contribution from the industry partner(s). We sincerely thank the anonymous reviewers for their valuable comments on this paper.
