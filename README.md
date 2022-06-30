[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) Labelline
===
This repository contains a line segment annotation tool for registered frame and event data, which is implemented with PyQt5.

## UI

<p align="center">
    <img src="UI.png"/>
</p> 

## Requirements

* python3
* PyQt5
* numpy, glob, cv2, PIL, scipy, argparse, yacs, logging

## Dataset structure

    |-- dataset
        |-- <image folder>  
            |-- 000001.png  
            |-- 000002.png  
            |-- ...  
        |-- <label folder>   
            |-- 000001.mat  
            |-- 000002.mat  
            |-- ...
    
## Usage
```
python Labelline.py
```
Please refer to [Usage.md](Usage.md) for more usage information.