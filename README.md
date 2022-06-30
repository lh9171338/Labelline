[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) Labelline
===
This repository contains a fisheye image line segment annotation tool (without the requirement of distortion coefficients), which is implemented with PyQt5.

## UI

<p align="center">
    <img src="UI.png"/>
</p> 

## Requirements

* python3
* PyQt5
* numpy, glob, cv2, PIL, scipy, yacs, logging

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