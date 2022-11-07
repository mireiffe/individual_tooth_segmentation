# Individual Tooth Segmentation in Human Teeth Images

## Summary

An implementation for individual tooth segmentation method in human teeth image taken outside oral cavity by an optical camera. 

| Ground truth | Proposed |
|:-------:|:-------:|
| <img src="figures/gt1.png" width="300"> | <img src="figures/gt1.png" width="300"> |


[CMI Lab.](http://parter.kaist.ac.kr/), Department of Mathematical Sciences, KAIST.

# Instruction

## Requirements

- Python >= 3.7
- Pytorch >= 1.9.0
- Opencv
- yaml

## Neural network parameters

You can download pretrained parameters as pytorch checkpoint file.

<http://parter.kaist.ac.kr/colee/work/segmentation22/CP_teeth_seg.pth>\
(You may have to copy & paste the link into the address bar.)

## Configuration file

Set path of root directory, checkpoint, and input images in [config](config/default.yaml).

## One-click tutorial

After downloading the pytorch checkpoint file, one can start quickly with following command using a sample [test image](dataset/samples/999999.png):

```
git clone https://github.com/mireiffe/individual_tooth_segmentation.git
cd individual_tooth_segmentation
python main.py --All
```


# Result

## Test images

- 10 optical teeth images taken outside oral cavity.
- In consultation with a dentist, we generated ground truths for the test images.

## Benchmark method

In this repository, we compare the proposed method with only one [benchmark method](https://ieeexplore.ieee.org/abstract/document/9065216) based on the Mask R-CNN. There are more benchmark methods and comparisons in [[1]](#references).

## Evaluation

- F1-score

    For ground truth $G$ and given region $R$,
    $$F1 = \frac{2|G\cap{R}|}{|G| + |R|}.$$

- The 12 front teeth

    We calculate F1-score for 12 front teeth and uses $mF1$ as the evaluation metric, which is defined as the average F1-score for the 12 front teeth.
    
    - Teeth number: 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, and 43\
    [(FDI World Dental Federation notation.)](https://en.wikipedia.org/wiki/FDI_World_Dental_Federation_notation)

-   Table 1. The $mF1$ values of 10 optical teeth images for the two methods, Mask R-CNN and ours. Each $mF1$ is followed by the number of teeth having F1-score > 0.8. The proposed method segments the largest number of teeth in all images. It also shows the best mF1 for all images except T05 and T10. Best scores and largest numbers are highlighted in bold.
    
    | Image ID | Mask R-CNN | Proposed |
    |:--------:|------|----------|
    | 01 | 0.9508 (**13**) | **0.9642** (**13**)|
    | 02 | 0.9515 (6)  | **0.9732** (**7**)|
    | 03 | 0.9435 (12) | **0.9509** (**16**)|
    | 04 | 0.7874 (11) | **0.9341** (**16**)|
    | 05 | **0.9411** (12) | 0.9346 (**17**)|
    | 06 | 0.6232 (8)  | **0.9326** (**15**)|
    | 07 | 0.9296 (6)  | **0.9655** (**8**)|
    | 08 | 0.6312 (4)  | **0.7910** (**5**)|
    | 09 | 0.9246 (13) | **0.9513** (**16**)|
    | 10 | **0.9432** (12) | 0.9325 (**18**)|

## Segmentation results

| Image ID | Ground truth | Mask R-CNN | Proposed |
|:-------:|:-------:|:-------:|:-------:|
| 01 | <img src="figures/gt0.png" width="300"> | <img src="figures/mrcnn_0.png" width="300"> | <img src="figures/proposed0.png" width="300"> |
| 01 | <img src="figures/gt1.png" width="300"> | <img src="figures/mrcnn_1.png" width="300"> | <img src="figures/proposed1.png" width="300"> |
| 02 | <img src="figures/gt2.png" width="300"> | <img src="figures/mrcnn_2.png" width="300"> | <img src="figures/proposed2.png" width="300"> |
| 03 | <img src="figures/gt3.png" width="300"> | <img src="figures/mrcnn_3.png" width="300"> | <img src="figures/proposed3.png" width="300"> |
| 04 | <img src="figures/gt4.png" width="300"> | <img src="figures/mrcnn_4.png" width="300"> | <img src="figures/proposed4.png" width="300"> |
| 05 | <img src="figures/gt5.png" width="300"> | <img src="figures/mrcnn_5.png" width="300"> | <img src="figures/proposed5.png" width="300"> |
| 06 | <img src="figures/gt6.png" width="300"> | <img src="figures/mrcnn_6.png" width="300"> | <img src="figures/proposed6.png" width="300"> |
| 07 | <img src="figures/gt7.png" width="300"> | <img src="figures/mrcnn_7.png" width="300"> | <img src="figures/proposed7.png" width="300"> |
| 08 | <img src="figures/gt8.png" width="300"> | <img src="figures/mrcnn_8.png" width="300"> | <img src="figures/proposed8.png" width="300"> |
| 09 | <img src="figures/gt9.png" width="300"> | <img src="figures/mrcnn_9.png" width="300"> | <img src="figures/proposed9.png" width="300"> |

Image files are available in the [figures](figures) directory.

## References
1. Kim, Seongeun and Lee, Chang-Ock, *Individual Tooth Segmentation in Human Teeth Images Using Pseudo Edge-Region Obtained by Deep Neural Networks,* http://dx.doi.org/10.2139/ssrn.4159811.

2. G. Zhu, Z. Piao, S. C. Kim, *Tooth detection and segmentation with mask R-CNN*, in: 2020 International Conference on Artificial Intelligence in Information and Communication (ICAIIC), 2020, pp. 70–72.
