# data path
../data_large/train/\*.jpg  
../data_large/test/\*.jpg  

# model path
./checkpoints/XX.pth  

# pytorch version
0.3.0.post4  
- pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl --user
- pip3 install --no-deps torchvision

# train and test command
./scripts/train_pix2pix.sh  
./scripts/test_pix2pix.sh  

