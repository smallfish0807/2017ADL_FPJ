# 2017ADL_FPJ
## Install packages
- python: 3.5.2  
- tensorflow: 1.4.1  
- pytorch: 0.3.0.post4
  - pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl --user
  - pip3 install --no-deps torchvision
- pillow: 4.3.0
- visdom: 0.1.7
- dominate: 2.3.1
- imageio: 2.2.0

## Data
Downsampled ImageNet: http://image-net.org/small/download.php

## How to execute

### pix2pix training
pix2pix/scripts/train_pix2pix.sh
### pix2pix testing
pix2pix/scripts/test_pix2pix.sh

### pixel-cnn++ training
python3.5 pixel-cnn/train.py
### pixel-cnn++ testing
python3.5 pixel-cnn/generate.py --load_params=save/params_imagenet.ckpt

## Plan
11/25 Survey paper + learn corresponding techniques  
12/02 data processing + implement baseline + try different evaluation metrics  
12/09 Experiments  
12/16 Implement different models + Progress report  
12/23 Implement different models  
12/30 Implement different models  
01/06 Experiments + Make posters  
01/13 Presentation  

## HACKMD
- FPJ proposal: https://hackmd.io/GwMwnADARgTAhhAtMAjAdgMaICwQKYiJQoQAciAJmscGiAKwrZ5hA===  
- FPJ progress report: https://hackmd.io/KYFmCMHYBMGYDYC0BOAhsAZokkAcIU4AmRI4eeABl3HhHg1yA===  
- FPJ poster: https://hackmd.io/GYEwzMwMZQbAtMArAIwBzwCzAIYCZ4dMBTATnmMwHZhY6BGEFYABiA==
- FPJ report: https://hackmd.io/MYdmCMFMEYBZYLTQJwAZUNgM3AJgQIYBskiqAJltOecuSgByxA==

## References
### Paper   
- [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759.pdf)  
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf)  
- [PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications](https://arxiv.org/pdf/1701.05517.pdf)
### Github  
- [carpedm20/pixel-rnn-tensorflow](https://github.com/carpedm20/pixel-rnn-tensorflow)  
- [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)  
- [openai/pixel-cnn](https://github.com/openai/pixel-cnn)
