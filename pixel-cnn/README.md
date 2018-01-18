# pixel-cnn++

## Dependencies
- python3
- tensorflow 1.4
- imageio

## Usage
python train.py

## Parameters you might want to change
- data_dir default='../data'
- save_dir default='../save'
- nr_gpu default=1
- gpu_memory default=0.5

## References
- [PixelCNN++ paper](https://openreview.net/pdf?id=BJrFC6ceg)
- [Original Implementation](https://github.com/openai/pixel-cnn)

## Citation
```
@inproceedings{Salimans2016PixeCNN,
  title={PixelCNN++: A PixelCNN Implementation with Discretized Logistic Mixture Likelihood and Other Modifications},
  author={Tim Salimans and Andrej Karpathy and Xi Chen and Diederik P. Kingma and Yaroslav Bulatov},
  booktitle={Submitted to ICLR 2017},
  year={2016}
}
```
