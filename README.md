# Keyhole absorptivity prediction using ConvNets

Prediction of laser absorptivity from synchrotron x-ray images using deep convolutional neural networks

## Data
1. [A-AMB2022-01 Benchmark Challenge Problems](https://www.nist.gov/ambench/amb2022-01-benchmark-challenge-problems)
2. [Asynchronous AM Bench 2022 Challenge Data: Real-time, simultaneous absorptance and high-speed Xray imaging](https://data.nist.gov/od/id/mds2-2525)

## Catalog
* Image preprocessing code
* DataSet code
* Reset50 and ConvNeXt_tiny training code
* Evaluation code

## Results 
| Model | if pretrained | train loss | val loss | test loss |
| --- | --- | --- | --- | --- |
| ResNet50 | pretrained=True | 0.2139 | 1.4044 | 4.5335 |
|  | pretrained=False | 0.4583 | 1.2890 | 8.1304 |
| ConvNeXt | pretrained=True | 0.2129 | 1.3232 | 5.9008 |
|  | pretrained=False | 4.4071 | 6.9980 | 22.0075 |

## Acknowledgement
This work is implemented using [ResNet50](https://github.com/KaimingHe/deep-residual-networks) and [ConvNeXt_tiny](https://github.com/facebookresearch/ConvNeXt).

The model interpretation deployed the [CAM](https://github.com/jacobgil/pytorch-grad-cam). 

