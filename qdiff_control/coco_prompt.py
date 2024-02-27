from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import json
import random
from random import shuffle
from PIL import Image

def get_prompts(json_file='/dataset/coco2014/annotations/captions_val2014.json'):
    
    list_prompts = []
    data=json.load(open(json_file,'r'))


    for ann in data['annotations']:
        list_prompts.append(ann['caption'])

    # min_height = 1024
    # min_width = 1024
    # i = 0
    # for image in data['images']:
    #     if image['height'] < min_height:
    #         min_height = image['height']
    #     if image['width'] < min_width:
    #         min_width = image['width']
    #     if image['height'] < 300 or image['width'] < 300:
    #         i = i+1
    # print(min_height)
    # print(min_width)
    # print(i)
    shuffle(list_prompts)
    return list_prompts

def center_resize_image(path_image, out_path, size):
    num = 0
    for filename in os.listdir(path_image):
#        print(filename)
        # 判断是否为图片文件
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.JPEG'):
            # 拼接文件路径
            file_path = os.path.join(path_image, filename)
            img = Image.open(file_path)
            if filename.endswith('.JPEG') and img.mode=='RGBA':
                continue
            width, height = img.size
            square = min(width, height)
            center_x = int(width)/2
            center_y = int(height)/2
            x1 = int((width - square)/2)
            y1 = int((height - square)/2)
            box = (x1, y1, x1+square, y1+square)
            img = img.crop(box)
            image=img.resize(size, resample=Image.BICUBIC)#, box=box

            out_image = os.path.join(out_path, filename)
            image.save(out_image)
            num =  num + 1
            if num % 5000 == 0:
                print(num)

if __name__ == "__main__":
    # json_file='/dataset/coco2014/annotations/captions_val2014.json' # # Object Instance 类型的标注
    # get_prompts(json_file)
    # path_image = "/dataset/coco2014/val2014/"
    # out_path = "/dataset/coco2014/val2014_resize/"
    path_image = "/dataset/imagenet/train1/"
    out_path = "/dataset/imagenet/train_new/"
    if os.path.exists(path_image) and os.path.exists(out_path): 
        print("文件存在") 
    else: 
        print("文件不存在")

    # path_image = "/home/liuxuewen/Dome/q-diffusion/imagetxt/"
    # out_path = "/home/liuxuewen/Dome/q-diffusion/imagetxt_resize/"
    center_resize_image(path_image, out_path, (256, 256))