****  
**A fast-running version：**
link: https://pan.baidu.com/s/1DzSTCFCwLmJ2BT0q104TXg   code：5abn      
Google:https://drive.google.com/file/d/1ZHgdRs-yUzreSFOThOP00MIUVxeRoMYc/view?usp=sharing    
****  

The source code of our paper of IEEE Transactions on Image Processing(V2):  

Exploring Rich and Efficient Spatial Temporal Interactions for Real Time Video Salient Object Detection  
===  
Update:  
---  
Because the code of V1 is relatively long to upload,   
we re-implemented STVS based on the BBSNet and uploaded it to Baidu Cloud Disk.   

link：https://pan.baidu.com/s/1tneKPmyvmMBPyv_meZmeiQ   
code：3dqb   

Note:  
---  
1. The code is mainly for the second overhaul of the paper.    
2. The training data needs to be enhanced (Interval 2, 3, 4, 5; Rotation; Mirroring; Gaussian noise; Lighting changes; Scale changes, etc.)  
3. At present, there is another way to realize that 3 frames of images correspond to 3 outputs, and then sum of the losses of the 3 frames.(still in progress)  

****
The source code of our manuscript submitted to IEEE Transactions on Image Processing(V1):   

Exploring Rich and Efficient Spatial Temporal Interactions for Real Time Video Salient Object Detection   
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
Results on VOS-test dataset:https://pan.baidu.com/s/1FbqHNlqP07BL5k0sg4Wyaw, extraction: 16ep. 

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

Training:
---
1. Using the entire training set to pretrain the spatial branch.
2. Finetuing the whole spatiotemporal model using all training set.
