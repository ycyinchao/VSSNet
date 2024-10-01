# Camouflaged Object Detection via Complementary Information-Selected Network based on Visual and Semantic Separation

![](./images/framework.jpg)



# 1. Abstract

> Camouflaged object detection (COD) is a promising yet challenging task that aims to segment objects concealed within intricate surroundings, a capability crucial for modern industrial applications. Current COD methods primarily focus on the direct fusion of highlevel and low-level information, without considering their differences and inconsistencies. Consequently, accurately segmenting highly camouflaged objects in challenging scenarios presents a considerable problem. To mitigate this concern, we propose a novel framework called visual and semantic separation network (VSSNet), which separately extracts low-level visual and high-level semantic cues and adaptively combines them for accurate predictions. Specifically, it features the information extractor module for capturing dimension-aware visual or semantic information from various perspectives. The complementary information-selected module leverages the complementary nature of visual and semantic information for adaptive selection and fusion. In addition, the region disparity weighting strategy encourages the model to prioritize the boundaries of highly camouflaged and difficult-to-predict objects. Experimental results on benchmark datasets show the VSSNet significantly outperforms State-of-the-Art COD approaches without data augmentations and multiscale training techniques. Furthermore, our method demonstrates satisfactory cross-domain generalization performance in real-world industrial environments.
>



# 2. Results

![](./images/result.jpg)

![](./images/vis1.jpg)

![](./images/vis2.jpg)



# 3. Preparations

## 3.1 Datasets



## 3.2 Create and activate conda environment

```bash
conda create -n VSSNet python=3.8
conda activate VSSNet

git clone https://github.com/ycyinchao/VSSNet.git
cd VSSNet

pip install -r requirement.txt
```



## 3.3 Download Pre-trained weights

The pretrained model is stored in [Google Drive](https://drive.google.com/file/d/1rKmp0Zu1ZL6Z9VsYfYAKRkG271AvZB6G/view?usp=sharing). After downloading, please put it in the './pretrained/' fold.



## 3.4 Train

```bash
python Train.py --train_path 'the path of TrainDataset' --test_path 'the path of TestDataset'
```

Our well-trained model is stored in [Google Drive](https://drive.google.com/file/d/1_iqEtc5VvhYSk5PSdyuFDsMdb7GlwdJo/view?usp=sharing), which should be moved into the fold './checkpoints/VSSNet_384/'.



## 3.5 Test

```bash
python MyTesting.py --pth_path 'the path of checkpoint'
```

The more qualitative results of VSSNet on four benchmarks (COD10K, NC4K, CAMO, CHAMELEON) have already been stored in [Google Drive](https://drive.google.com/file/d/1RV12SAH93VbAvrOw7zghJUgMrrklihj8/view?usp=sharing), please unzip it into the fold './results/'.



## 3.6 Eval

```bash
python test_metrics.py
```

the results of evaluation are also in [Google Drive](https://drive.google.com/file/d/1RV12SAH93VbAvrOw7zghJUgMrrklihj8/view?usp=sharing).



# 4. Citation

Please kindly cite our paper if you find it's helpful in your work.

```
@article{yin2024camouflaged,
  title={Camouflaged Object Detection via Complementary Information-Selected Network Based on Visual and Semantic Separation},
  author={Yin, Chao and Yang, Kequan and Li, Jide and Li, Xiaoqiang and Wu, Yifan},
  journal={IEEE Transactions on Industrial Informatics},
  year={2024},
  publisher={IEEE}
}
```



# 5. License

The source code is free for research and education use only. Any commercial use should get formal permission first.

