This is a Github repository for ESRGAN. I loaded pretrained weights from https://github.com/xinntao/Real-ESRGAN and implemented it. I do have a self trained model as well but it is only to be used for images with no darkness. Load all the required images for training and use the following commands in order to get super resolution images on google colab: 
1.	!git clone https://{token}@github.com/sachothegreat/AI_3.git
2.	from getpass import getpass token = getpass('authentication token’)
3.	!git clone https://github.com/xinntao/Real-ESRGAN.git
4.	%cd Real-ESRGAN
5.	!python -m pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url https://download.pytorch.org/whl/cu118
6.	!pip install basicsr
7.	!pip install -r requirements.txt
   !python setup.py develop
8.	!wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P weights
9.	!zip -r /content/AI_3_results.zip /content/AI_3/results

Acknowledgments
This project utilizes resources from the following:

1. Weights: The pre-trained weights are sourced from the Real-ESRGAN repository, licensed under the BSD 3-Clause License. For more details, refer to their license file: https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE.
2. Images: The images used for this project are from the SRGAN Image Super-Resolution project on Kaggle. Please refer to the dataset's Kaggle page for specific licensing and usage details: https://www.kaggle.com/code/minawagihsmikhael/srgan-image-super-resolution-pytorch/input.







