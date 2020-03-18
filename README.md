The source code of our manuscript submitted to IEEE Transactions on Image Processing:

A Multi-scale Spatiotemporal Network for Real-time Video Saliency Detection  
===

Prerequisites:
---
* CUDA v8.0, cudnn v7.0
* python 3.5
* pytorch 0.4.1
* torchvision
* numpy
* Cython
* GPU: NVIDIA GeForce GTX 1080 Ti

SaliencyMaps:
---
All results on Davis-T, SegTrack-V2, Visal, FBMS-T and DAVSOD-T datasets of our method are availabled from
baidu cloud: https://pan.baidu.com/s/1J9yYBeMXmaUvGQ1aAbBUYA, extraction: 45gi.  

TrainedModel:  
---
You can download the trained model from baidu cloud: https://pan.baidu.com/s/16wPfMNPjDlnwWx4xuM8R3Q,  
extraction: 7w7p.  

Usage:
---
a. Please first download the model.  
b. Please put test images under .\DataSet\.  
d. Please put the model under folder .\model\.  
e. Run demo.py.  
f. Results can be found in .\resutls\.  
