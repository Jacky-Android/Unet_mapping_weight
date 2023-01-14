# U-Net(Convolutional Networks for Biomedical Image Segmentation)

## 该项目主要参考以下开源仓库
* [https://github.com/milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)
* [https://github.com/pytorch/vision](https://github.com/pytorch/vision)
## 拥有unet版本：
* [unet](https://github.com/Jacky-Android/unet-torch-/blob/main/src/unet.py)
* [vgg16unet](https://github.com/Jacky-Android/unet-torch-/blob/main/src/vgg_unet.py)
* [convnext_unet](https://github.com/Jacky-Android/unet-torch-/blob/main/src/unet_convnext.py)
* [segnet](https://github.com/Jacky-Android/unet-torch-/blob/main/src/segnet.py)
* [unet_cbam](https://github.com/Jacky-Android/unet-torch-/blob/main/src/unet_sc.py)
* [unet_se](https://github.com/Jacky-Android/unet-torch-/blob/main/src/unet_selayer.py)
* [mobilenet_uet](https://github.com/Jacky-Android/unet-torch-/blob/main/src/mobilenet_unet.py)
* [unet_networks(unet_attention,R2UNet,R2UNet_attention)](https://github.com/Jacky-Android/unet-torch-/blob/main/src/unet_networks.py)
* [unet_eca](https://github.com/Jacky-Android/unet-torch-/blob/main/src/unet_eca.py)
## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.10
* Ubuntu或Centos(Windows暂不支持多GPU训练)
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`
## unet的提出，但是unet还有很大的改进可能
![image](https://img-blog.csdnimg.cn/20210316213927771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTA3NDU2OA==,size_16,color_FFFFFF,t_70)
## 文件结构：
```
  ├── src: 搭建U-Net模型代码
  ├── train_utils: 训练、验证以及多GPU训练相关模块
  ├── my_dataset.py: 自定义dataset用于读取DRIVE数据集(视网膜血管分割)
  ├── train.py: 以单GPU为例进行训练
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
  └── compute_mean_std.py: 统计数据集各通道的均值和标准差
```

