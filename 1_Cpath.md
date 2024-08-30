# Capth学习笔记
## 1.安装Qupath和下载需要的全幅切片图像

下载QuPath：
https://qupath.readthedocs.io/en/0.4/docs/intro/installation.html#download-install
![img1](data/img/image1.png)
从这里下载全幅切片图像：

https://drive.google.com/file/d/1GWtuOLRf6-7C8zCG6b3JdYDPk_C5IS3_/view?usp=sharing
https://drive.google.com/file/d/1o9Es1qdY15y8q_gsjX3KVkn7HS8amlSF/view?usp=sharing
使用QuPath探索这两个全幅切片图像（WSIs）：

一张带有坏死区域的大鼠肝脏尸检图像
一张侵袭性乳腺癌（BRCA）的TCGA样本图像

## 2.OpenSlide安装

    # Install openslide 
    !pip install openslide-python

这里未安装pandas和matplotlib，需要安装： 

    pip install pandas    
    pip install matplotlib

### Accessing slide properties and reading

    # Import openslide
    # 理解幻灯片属性 

    import os
    import pandas as pd
    from glob import glob
    from pprint import pprint
    import matplotlib.pyplot as plt 
    from openslide import OpenSlide



### 遇到错误
读取openslide包出错  
    
    from openslide import OpenSlide


遇到的错误是由于 Python 的 openslide 库无法找到必需的动态链接库 (libopenslide-0.dll 或 libopenslide-1.dll)，这些文件是 OpenSlide 在 Windows 上运行所必须的。

解决方法：从openslide官网（ https://openslide.org/ ）安装对应的openslide库。
在 Windows 上，下载 OpenSlide Windows 二进制文件并提取它们 拖动到已知路径。然后，在语句中导入：openslidewith os.add_dll_directory()

    #The path can also be read from a config file, etc.
    OPENSLIDE_PATH = r'c:\path\to\openslide-win64\bin'

    import os
    if hasattr(os, 'add_dll_directory'):
        # Windows
        with os.add_dll_directory(OPENSLIDE_PATH):
            import openslide
    else:
        import openslide

## 3.Accessing slide properties and reading

    slide_path = os.path.join('data', 'slides', 'necrosis.tiff')
    slide = OpenSlide(slide_path)

    # 检查最高分辨率（20x）下的图像尺寸： 
    print('图像尺寸为：', slide.dimensions[0], ' x ', slide.dimensions[1])

    # 检查预提取级别的数量：
    print('级别数量：', slide.level_count)

    # 检查每个级别的尺寸：
    print('所有级别：')
    for dims in slide.level_dimensions:
        print('    - (', dims[0], ' x ', dims[1], ')')

    # 检查幻灯片大小 
    print(f"文件大小：{round(os.path.getsize(slide_path) / 1024 ** 2, 2)} MB")

    # 可选：检查所有幻灯片属性 
    # pprint(dict(slide.properties))

输出：

    图像尺寸为： 61751  x  35769
    级别数量： 3
    所有级别：
        - ( 61751  x  35769 )
        - ( 15437  x  8942 )
        - ( 3859  x  2235 )
    文件大小：671.22 MB

## 4.Choosing appropriate magnification levels and patch size
![img1](data/img/image2.png)

_转载自 https://aletolia.github.io/Session%201/_