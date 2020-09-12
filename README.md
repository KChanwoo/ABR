# ABR

ABR flask server
1. save image to server
2. extract graph, find peak
3. batch to extract train data
4. train yolov4, RNN (and using)


## Environment
```
python 3.7 ~
opencv-python lastest
tensorflow 2.3 ~
flask lastest
```

## Install
```
git clone https://github.com/Yonsei-Maist/ABR-image-processor.git
```

## Run
```
# server
python3 serve.py

# batch
python3 batch.py

# set configure
vi config.py (need restart)
```

## Author
Chanwoo Gwon, Yonsei Univ. Researcher since 2020.05

## Maintainer
Chanwoo Gwon, arknell@yonsei.ac.kr (2020.09 ~)
