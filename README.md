# 메뉴판 이미지 인식을 통해 text로 변환  

## CRAFT: Text detection을 위한 글자 영역 인식

### Overview
![image](https://user-images.githubusercontent.com/80943639/145672300-435f6f82-73d0-40ff-bd8b-34acbec058f1.png)

CRAFT text detector는 각 글자의 위치와 근처의 글자들과의 affinity를 찾아서 text bounding box를 만들어주는 pytorch 모델이다.
Text의 boundung box는 글자 구역과 affinity 점수를 이용해 thresholding한 후 가장 작은 bounding box를 return하는 형태로 만들어진다.
우리는 메뉴를 잘 인식하도록 threshold를 조정하여 얻어진 bounding box 부분을 crop하여 이미지 파일로 저장하도록 코드를 변형하였다.

## Getting started
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
