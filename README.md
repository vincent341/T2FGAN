# T2FGAN
--------------------- 
by Shuang Liu, Mete Ozay, Hongli Xu, Yang Lin, Okatani Takayuki
### Table of Contents
0. [Introduction](#Introduction)
0. [Prerequists](#Prerequists)
0. [Models](#Models)
0. [Running](#Running)
0. [Contact](#Contact)
## Introduction
This is an implementation of our work T2FGAN. T2FGAN is able to generate images of underwater active landmarks with arbitary particular water quality, illumination, pose and landmark configurations (WIPCs). Some codes are borrowed from [pix2pix-tensorflow](https://github.com/affinelayer/pix2pix-tensorflow).

## Prerequisites
  - Python 3.5
  - tensorflow 1.9.0
  - opencv 3.4
  - shapely
## Models 
The trained model can be downloaded [here](http://vision.is.tohoku.ac.jp/~liushuang/tank2fieldGAN/model/).
## Running
0. The model files provided in [Models](#Models) are neccesary for running. Unzip the model folder and copy the folder to "$currentpath/model/". 
1. The file "tank2fieldgan.py" is the main file for running T2FGAN. An example is given for running "tank2fieldgan.py" below. Details of the used parameters are given in the table below. Generated images are saved in "generated_img/".
```
python3 tank2fieldgan.py --mode test --output_dir $currentpath/res --input_dir ./ --checkpoint $currentpath/model/model_newarch4step0.0002_e1200/ --heading 0.0 --pitch 0.0 --roll 0.0 --Tx 0.0 --Ty 0.0 --Tz 8000 --lx -40 --ly 150 --lz 300 --betar 0.3 --betag 0.003 --betab 0.026 --light 150 --g 0.68
```
| Parameters        | Descripitons |
| ------------- |:-------------:|
| ( heading, pitch, roll )      | The roation between the camera and underwater landmarks  represented by Euler angles.|
| (Tx, Ty, Tz)      | The translation vector between the camera and underwater landmarks.       |
| (lx,ly, lz) | The position of the light source.      |
|  (betar, betag, betab) | Extinction coefficient for R,G and B channel respectively.  | 
| light | Light intensity. |
| g | The parameter of the phase function. |
## Contact
If you encounter any problem in running this project, please feel free to contact via email. Email: liushuangvision@gmail.com

