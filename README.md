# Implementation of Deep-Learning model extracting text in menu using text detection and text recognition  

## Overview
![image](https://user-images.githubusercontent.com/80943639/145672300-435f6f82-73d0-40ff-bd8b-34acbec058f1.png)

## Text recognition 모델
Text Recognition Model은 300,000번의 Epoch로 한국어 글자체 이미지를 pretraining한 모델을 사용하였다. Traing 결과 Test Accuracy는 88%에 달했고 한국어 뿐만 아니라 영어와 숫자 또한 인식할 수 있다는 점에서 유효한 결과를 보였다. 

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

### Demo 실행
* RecognitionModel에서 demo 실행
```
import shutil
!python ../DetectionModel/test.py --trained_model=../DetectionModel/craft_mlt_25k.pth --test_folder=../cropImages/237

opt = demoOpt()

if executeDemo == True:
    if opt.sensitive:
        opt.character = string.printable[:-6]

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    predLists = []

    fileNum = len(os.listdir("./demoImages"))

    for i in range(fileNum):
        opt.image_folder = "./demoImages/" + os.listdir("./demoImages")[i]
        predLists.append(demo(opt))
    
    for i in range(fileNum):
        dir_path = "./demoImages/" + os.listdir("./demoImages")[0]
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    
    for x in predLists:
        predicted_string = ""
        for y in x:
            predicted_string += y +' '
        print(predicted_string)
```
두 번째 행 맨 뒤의 숫자만 바꾸어서 demo 실행
e.g. --test_folder=../crepImages/3

## Text Detection 모델: CRAFT Text detector
CRAFT text detector는 각 글자의 위치와 근처의 글자들과의 affinity를 찾아서 text bounding box를 만들어주는 pytorch 모델이다.
Text의 boundung box는 글자 구역과 affinity 점수를 이용해 thresholding한 후 가장 작은 bounding box를 return하는 형태로 만들어진다.
우리는 메뉴를 잘 인식하도록 threshold를 조정하여 얻어진 bounding box 부분을 crop하여 이미지 파일로 저장하도록 코드를 변형하였다.

### Pretrained 모델을 이용하여 test 진행

* pretrained model 실행
``` (with python 3.7)
python test.py --trained_model=[weightfile] --test_folder=[folder path to test images]
```
결과 이미지는 './result' 에 저장

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

