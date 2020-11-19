# From Semantic Categories to Fixations: A Novel Weakly-supervised Visual-auditory Saliency Detection Approach.
## dependencies
## preparation
1.download the official pretrained model [model]net = torch.hub.load('facebookresearch/WSL-Images','resnext101_32x8d_wsl') of ResNeXt implemented in Pytorch if you want to train the network again.
2.download or put the AVE datasets, AVAD, DIEM, SumMe, ETMD, Coutrot1, Coutort2(Google drive) in the folder of data for training or test.
## training
stage1.Scoarse, STcoarse, SAcoarse
stage2.
stage3.
you may revise the TAG and SAVEPATH defined in the train.py. After the preparation, run this command
python train.py
## testing
After the preparation, run this commond
python test.py
We provide the trained model file (Google drive), and run this command to check its completeness:
The saliency maps are also available (Google drive).
## evaluation
We provide the evaluation code in the folder "eval_code" for fair comparisons. 
You may need to revise the algorithms , data_root, and maps_root defined in the main.m. 
The saliency maps of the competitors are provided (Google drive).
Quantitative comparisons:
![Quantitative](https://github.com/CVPR2021Submit/STANet/blob/main/fig/cvpr2021.gif)
Qualitative comparisons:
