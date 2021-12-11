# Implementation of Deep-Learning model extracting text in menu using text detection and text recognition  

## Overview
![image](https://user-images.githubusercontent.com/80943639/145672300-435f6f82-73d0-40ff-bd8b-34acbec058f1.png)

## Text recognition 모델

### Hyperparameters
- `Loss Criterion`: CTCLoss
- `Batch_size`: 96

- `Epochs`: 300,000 

- `Validation`: 1회 / 10,000 epochs

- `Evaluation Dataset`: ‘IIIT5k_3000’, ‘SVT’ 등 유명한 10가지의 dataset

- `Confidence Score`: Text값이 얼마나 신뢰할 수 있는 지 Training과 Evaluation에서 모두 계산

### Confidence Score 계산 코드
```
try:
    confidence_score = pred_max_prob.cumprod(dim=0)[-1]
except:
    confidence_score = 0 
    confidence_score_list.append(confidence_score)
```
numpy의 cumprod를 사용하여 pred_max_prob 변수의 누적 곱들로 반환하였다



## Text Detection 모델: CRAFT Text detector
CRAFT text detector는 각 글자의 위치와 근처의 글자들과의 affinity를 찾아서 text bounding box를 만들어주는 pytorch 모델이다.
Text의 boundung box는 글자 구역과 affinity 점수를 이용해 thresholding한 후 가장 작은 bounding box를 return하는 형태로 만들어진다.
우리는 메뉴를 잘 인식하도록 threshold를 조정하여 얻어진 bounding box 부분을 crop하여 이미지 파일로 저장하도록 코드를 변형하였다.

### Install dependencies
#### Requirements
- PyTorch>=0.4.1
- torchvision>=0.2.1
- opencv-python>=3.4.2
- check requiremtns.txt
```
pip install -r requirements.txt
```

### Pretrained 모델을 이용하여 test 진행

* Run with pretrained model
``` (with python 3.7)
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```
결과 이미지는 './result' 에 저장된다.

### Arguments
* `--trained_model`: pretrained model (craft_mlt_25k.pth)
* `--text_threshold`: text confidence threshold
* `--low_text`: text low-bound score
* `--link_threshold`: link confidence threshold
* `--cuda`: use cuda for inference (default:True)
* `--canvas_size`: max image size for inference
* `--mag_ratio`: image magnification ratio
* `--poly`: enable polygon type result
* `--show_time`: show processing time
* `--test_folder`: folder path to input images
* `--refine`: use link refiner for sentense-level dataset
* `--refiner_model`: pretrained refiner model

## Data path 
- AI허브 개방 데이터 - 비전 - 한국어 글자체 이미지 - Text in the wild - 01_textinthewild_goods_images_new
- AI허브 개방 데이터 - 국토환경 - 관광 지식베이스 - 한국 관광 POI데이터셋 - 크롤링_메뉴판

