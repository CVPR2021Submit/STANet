# From Semantic Categories to Fixations: A Novel Weakly-supervised Visual-auditory Saliency Detection Approach.  
![net](https://github.com/CVPR2021Submit/STANet/blob/main/fig/net.gif)  
## Abstract
Thanks to the rapid advances in the deep learning techniques and the wide availability of large-scale training sets, the performances of video saliency detection models have been improving steadily and significantly. However, the deep learning based visual-audio fixation prediction is still in its infancy. At present, only a few visual-audio sequences have been furnished with real fixations being recorded in the real visual-audio environment. Hence, it would be neither efficiency nor necessary to re-collect real fixations under the same visual-audio circumstance. To address the problem, this paper advocate a novel approach in a weaklysupervised manner to alleviating the demand of large-scale training sets for visual-audio model training. By using the video category tags only, we propose the selective class activation mapping (SCAM), which follows a coarse-to-fine strategy to select the most discriminative regions in the spatial-temporal-audio circumstance. Moreover, these regions exhibit high consistency with the real human-eye fixations, which could subsequently be employed as the pseudo GTs to train a new spatial-temporal-audio (STA) network. Without resorting to any real fixation, the performance of our STA network is comparable to that of the fully supervised ones.  
## Dependencies
* python 3.6  
* pytorch 1.2.0  
* soundfile  
## Preparation
1.Download the official pretrained model 

  net = torch.hub.load('facebookresearch/WSL-Images','resnext101_32x8d_wsl')
of ResNeXt implemented in Pytorch, and vggsound model net = torch.load('vggsound_netvlad')  if you want to train/test the network.  
2.Download and put the AVE datasets, AVAD, DIEM, SumMe, ETMD, Coutrot1, Coutort2(Google drive) in the folder of data for training or test.  
## Training
- Stage 1. Train the model of S<sub>coarse</sub>, ST<sub>coarse</sub>, SA<sub>coarse</sub> respectively.  
- Stage 2. Train the model of S<sub>fine</sub>, ST<sub>fine</sub>, SA<sub>fine</sub> respectively.   
- Stage 3. STANet  
python train.py  
## Testing 
After the preparation, run this commond  
python test.py  
We provide the trained model file (Google drive), and run this command to check its completeness:  
The saliency maps are also available (Google drive).  
## Evaluation
We provide the evaluation code in the folder "eval_code" for fair comparisons.   
You may need to revise the algorithms , data_root, and maps_root defined in the main.m.   
The saliency maps of the competitors are provided (Google drive).  
Quantitative comparisons:  
![Quantitative](https://github.com/CVPR2021Submit/STANet/blob/main/fig/cvpr2021.gif)  
Qualitative comparisons:  
