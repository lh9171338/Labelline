[<img height="23" src="https://raw.githubusercontent.com/lh9171338/Outline/master/icon.jpg"/>](https://github.com/lh9171338/Outline) Labelline
===

# Introduction
This repository contains a line segment annotation tool for pinhole, fisheye, and spherical images, which is implemented with PyQt5.

# UI

<p align="center">
    <img width="100%" src="UI.png"/>
</p> 

# Requirements

```shell
pip install -r ./requirements.txt
```

## Dataset structure

```shell
|-- dataset   
    |-- <image folder>
        |-- 000001.png  
        |-- 000002.png  
        |-- ...  
    |-- <label folder>  
        |-- 000001.mat  
        |-- 000002.mat  
        |-- ...  
    |-- <coeff folder>
        |-- 000001.yaml
        |-- 000002.yaml  
        |-- ...
```

## Usage
```
python Labelline.py --type <image type> [--coeff_file <coeff image>]
```
Please refer to [Usage.md](Usage.md) for more usage information.