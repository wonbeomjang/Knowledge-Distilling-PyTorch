# Distilling the Knowledge in a Neural Network
Implementation of Distilling the Knowledge in a Neural Network https://arxiv.org/pdf/1503.02531.pdf

## Accuracy
### cifar 10

#### Teacher Accuracy
|   vgg11  | ResNet18 | ResNet50 | ResNet101 |
|:--------:|:--------:|:--------:|:---------:|
| 82.6078% | 89.5004% | 87.7389% |  88.3260% |

#### distillation accuracy
|      Model    | Accuracy |          |
|:-------------:|:--------:|:--------:|
|   base line   | 70.6669% |          |
|  + VGG 11 KD  | 72.2943% |          |
| + ResNet18 KD | 75.1597% |          |
| + ResNet50 KD | 76.1282% |          |
| + ResNet101 KD| 75.9685% |          |

### cifar 100

#### Teacher Accuracy
|   vgg19  | ResNet18 | ResNet50 | ResNet101 |
|:--------:|:--------:|:--------:|:---------:|
| 35.7628% | 61.8332% | 57.9319% |  55.4014% |

#### distillation accuracy
|      Model    | Accuracy |          |
|:-------------:|:--------:|:--------:|
|   base line   | 37.1206% |          |
|   + self KD   | 38.2887% |          |
|  + VGG 19 KD  | 37.8494% |          |
| + ResNet18 KD | 43.7899% |          |
| + ResNet50 KD | 57.9319% |          |
| + ResNet101 KD| 46.7252% |          |