[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) Labelline
===
This repository contains a line segment annotation tool for pinhole, fisheye, and spherical images, which is implemented with PyQt5.

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
        |-- data-01  
            |-- images  
                |-- 000001.png  
                |-- 000002.png  
                |-- ...  
            |-- labels  
                |-- 000001.mat  
                |-- 000002.mat  
                |-- ...  
            |-- coeffs
                |-- 000001.yaml
                |-- 000002.yaml  
                |-- ...  
        |-- data-02  
        |-- ...

    or

    |-- dataset  
        |-- data-01
            |-- 000001.png  
            |-- 000001.mat  
            |-- 000001.yaml
            |-- 000002.png  
            |-- 000002.mat  
            |-- 000002.yaml
            |-- 000003.png  
            |-- 000003.mat  
            |-- 000003.yaml
            |-- ...
        |-- data-02  
        |-- ...

## Usage
```
python Labelline.py --type <image type> [--coeff_file <coeff image>]
```
