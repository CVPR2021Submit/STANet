# From Semantic Categories to Fixations: A Novel Weakly-supervised Visual-auditory Saliency Detection Approach.
## dependencies
## preparation
1.download the official pretrained model (Google drive) of ResNet implemented in Pytorch if you want to train the network again.
2.download or put the RGB saliency benchmark datasets (Google drive) in the folder of data for training or test.
## training
you may revise the TAG and SAVEPATH defined in the train.py. After the preparation, run this command

python3 train.py
make sure that the GPU memory is enough (the original training is conducted on a NVIDIA RTX (24G) card with the batch size of 32).
## testing
After the preparation, run this commond

 python3 test.py model/model-xxxxx.pt
We provide the trained model file (Google drive), and run this command to check its completeness:

cksum model-100045448.pt 
you will obtain the result 100045448 268562671 model_100045448.pt. The saliency maps are also available (Google drive).
## evaluation
We provide the evaluation code in the folder "eval_code" for fair comparisons. 
You may need to revise the algorithms , data_root, and maps_root defined in the main.m. 
The saliency maps of the competitors are provided (Google drive).
Quantitative comparisons:
![Quantitative](https://github.com/CVPR2021Submit/STANet/blob/main/fig/cvpr2021.png)
Qualitative comparisons:
