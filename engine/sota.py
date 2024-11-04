import os
import sys
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "SOTA", "RMT-TSE","generators","UNIT"))
from utils import get_config, pytorch03_to_pytorch04
from trainer import MUNIT_Trainer, UNIT_Trainer
import argparse
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image

import cv2
import sys
import argparse
import numpy as np
import os
import json
from scipy import ndimage
def day2rain(dataset_path, output_path):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    config_file = os.path.join(parent_dir, "SOTA", "RMT-TSE","generators","UNIT","configs","unit_day2rain.yaml")

    config = get_config(config_file)
    trainer = UNIT_Trainer(config)
    trainer.cuda()
    trainer.eval()

    # state_dict = torch.load(dir_path + '/models/gen.pt')
    state_dict = torch.load(os.path.join(parent_dir, "SOTA", "gen_00030000.pt"))

    trainer.gen_a.load_state_dict(state_dict['a'])
    trainer.gen_b.load_state_dict(state_dict['b'])


    encode = trainer.gen_a.encode
    style_encode = trainer.gen_b.encode
    decode = trainer.gen_b.decode
    new_size = config['new_size']

    with torch.no_grad():
        transform = transforms.Compose([transforms.Resize(new_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_dir = dataset_path
        image_list = os.listdir(input_dir)
        for i, img_name in enumerate(image_list):
            if i < len(image_list):
                img = os.path.join(input_dir, img_name)
                if 'png' in img or 'jpg' in img:
                    image = Image.open(img).convert('RGB')
                    im_size = image.size
                    if im_size == (1920, 1208):
                        image = image.crop((0, 248, im_size[0], im_size[1]))
                    image = Variable(transform(image).unsqueeze(0).cuda())

                    content, _ = encode(image)
                    outputs = decode(content)
                    outputs = (outputs + 1) / 2.
                    path = os.path.join(output_path, img_name)
                    vutils.save_image(outputs.data, path, padding=0, normalize=True)


def gen_rain(input_path, output_path, folder=2):
    # if not os.path.exists(os.path.join(output_path, 'source_datasets')):
    #     os.makedirs(os.path.join(output_path, 'source_datasets'))

    # if not os.path.exists(os.path.join(output_path, 'follow_up_datasets')):
    #     os.makedirs(os.path.join(output_path, 'follow_up_datasets'))

    source_path = input_path
    img_list = os.listdir(source_path)
    for img_name in img_list:
        if '.png' in img_name:
            img = cv2.imread(os.path.join(source_path, img_name))
            # cv2.imwrite(os.path.join(output_path, 'x_n1', img_name), img)
            noise = get_noise(img, value=200)
            rain = rain_blur(noise, length=30, angle=-30, w=3)
            rain_img = alpha_rain(rain, img, beta=0.6)  # 方法一，透明度賦值
            cv2.imwrite(os.path.join(output_path,  img_name), rain_img)


def get_noise(img, value=10):
    noise = np.random.uniform(0, 256, img.shape[0:2])

    v = value * 0.01
    noise[np.where(noise < (256 - v))] = 0

    k = np.array([[0, 0.1, 0],
                  [0.1, 8, 0.1],
                  [0, 0.1, 0]])

    noise = cv2.filter2D(noise, -1, k)

    '''cv2.imshow('img',noise)
    cv2.waitKey()
    cv2.destroyWindow('img')'''
    return noise


def rain_blur(noise, length=10, angle=0, w=1):
    trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
    dig = np.diag(np.ones(length))
    k = cv2.warpAffine(dig, trans, (length, length))
    k = cv2.GaussianBlur(k, (w, w), 0)

    blurred = cv2.filter2D(noise, -1, k)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    '''
    cv2.imshow('img',blurred)
    cv2.waitKey()
    cv2.destroyWindow('img')'''

    return blurred


def alpha_rain(rain, img, beta=0.8):
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

    rain_result = img.copy()
    rain = np.array(rain, dtype=np.float32)
    rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
    rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
    rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]

    # cv2.imshow('rain_effct_result',rain_result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return rain_result


def add_rain(rain, img, alpha=0.9):
    # chage rain into  3-dimenis

    rain = np.expand_dims(rain, 2)
    rain = np.repeat(rain, 3, 2)

    result = cv2.addWeighted(img, alpha, rain, 1 - alpha, 1)
    cv2.imshow('rain_effct', result)
    cv2.waitKey()
    cv2.destroyWindow('rain_effct')

def add_person(rain, img, alpha=0.9):
    # chage rain into  3-dimenis

    rain = np.expand_dims(rain, 2)
    rain = np.repeat(rain, 3, 2)

    result = cv2.addWeighted(img, alpha, rain, 1 - alpha, 1)
    cv2.imshow('rain_effct', result)
    cv2.waitKey()
    cv2.destroyWindow('rain_effct')


def create_mask(image, threshold=30):  # 添加阈值参数
    """
    创建mask，将接近黑色的像素排除

    参数:
    image: PIL Image对象
    threshold: 判断接近黑色的阈值（0-255），越大包含的颜色越多
    """
    # 转换图片为RGB模式
    img = image.convert("RGB")
    data = np.array(img)

    # 创建接近黑色像素的mask
    # 检查RGB三个通道是否都小于阈值
    mask = (data[:, :, :3] > threshold).any(axis=2)

    # 使用膨胀和腐蚀操作来填充内部的小区域
    mask = ndimage.binary_dilation(mask)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask)

    # 转换mask为图像
    return Image.fromarray((mask * 255).astype(np.uint8))
from rembg import remove
def add_person(input_path,mask_path, output_path, source_image_path):
    with open(os.path.join("SOTA","Images", "results", "info.json"), 'r') as file:
        locations = json.load(file)
    source_path = input_path
    img_list = os.listdir(source_path)
    reference_image =  Image.open(source_image_path)
    reference_mask = create_mask(reference_image)
   # reference_mask = remove(reference_image, only_mask=True)

    original_width, original_height = reference_image.size
    cont = 35
    CCont = 0
    for img_name in img_list:
        if '.png' in img_name:
            img = Image.open(os.path.join(source_path, img_name))
            current_width, current_height = img.size

            bbox = locations[cont]['white_bbox']
            bbox_width = bbox[2] - bbox[0]

            CCont+=1
            if CCont %5==0:
                cont = cont-1
                if cont<10:
                    cont =35
            scale_factor = bbox_width / original_width
            new_height = int(original_height * scale_factor * 1)
            resized_reference_image = reference_image.resize((bbox_width, new_height), Image.LANCZOS)
            resized_reference_mask = reference_mask.resize((bbox_width, new_height), Image.LANCZOS)

            paste_x = int(bbox[0] * (current_width / 1920))  # 假设原始坐标基于1024x1024的图像
            paste_y = int(bbox[1] * (current_height / 1208))
            #mask = cv2.imread(os.path.join(mask_path, img_name))
            img.paste(resized_reference_image, (paste_x, paste_y), mask=resized_reference_mask)
            img.save(os.path.join(output_path, img_name))


def Create_roadside_images(save_dir,save_dir_1,reference_image_path,reference,seg_dir):
    with open(os.path.join(reference,"results","info.json"), 'r') as file:
        locations = json.load(file)
    with open(os.path.join(save_dir,"speed_segments.json"), 'r') as f:
        segments = json.load(f)
    reference_image = Image.open(reference_image_path)
    reference_mask = create_mask(reference_image)
    original_width, original_height = reference_image.size
    save_dir_2 = os.path.join(save_dir_1, "MT")
    save_dir_3 = os.path.join(save_dir_1, "Raw")
    data_process.Check_file(save_dir_2)
    data_process.Check_file(save_dir_3)
    for i in range(len(segments)):
        sampled_bboxes = uniform_sample_dicts(locations, len(segments[i]['Image File']))
        for j in range(len(segments[i]['Image File'])):
            road_mask = os.path.join(seg_dir,"center",os.path.basename(segments[i]['Image File'][j]))

            result =find_rightmost_edge_polygon(road_mask)
            #road_mask = cv2.imread(road_mask, cv2.IMREAD_GRAYSCALE)
            if result is None or len(result) == 0:
                continue
            bbox = scale_bbox(sampled_bboxes[j])

            bbox_width = bbox[2] - bbox[0]
            scale_factor = bbox_width / original_width
            new_height = int(original_height * scale_factor*0.9)
            resized_reference_image = reference_image.resize((bbox_width, new_height), Image.LANCZOS)
            resized_reference_mask = reference_mask.resize((bbox_width, new_height), Image.LANCZOS)

            #r#oad_mask = cv2.imread(road_mask)
            start_row = bbox[-1]
            # 找到最接近 start_row 的点
            # 找到最接近start_row的点
            rightmost_points = min(result, key=lambda p: abs(p[1] - start_row))

            # 在那一行上找最右边的点
            rightmost_points = find_rightmost_point(road_mask, rightmost_points)

            image = Image.open(segments[i]['Image File'][j])
            scaled_object = image.copy()
            if j==0:
                point_x = rightmost_points[0]
                point_y = rightmost_points[1]-new_height//2
                past_point_x =point_x
                past_point_y = point_y
            else:
                point_x = rightmost_points[0]
                point_y = rightmost_points[1] - new_height//2
                if abs(point_x-past_point_x)>50:
                    point_x = past_point_x
                    point_y = past_point_y
            road_mask_array = cv2.imread(road_mask, cv2.IMREAD_GRAYSCALE)

            while point_x > 0:
                road_area = road_mask_array[point_y:point_y + new_height, point_x:point_x + bbox_width]
                #if not np.any(road_area == 6):  # 假设6表示道路
                if not np.any((road_area == 6) | (road_area == 20) | (road_area == 102)):
                    break
                point_x += 1
            image.paste(resized_reference_image, (point_x, point_y), mask=resized_reference_mask)
            if point_x>image.width-bbox_width//2 or point_y<0:
                pass
            else:
                save_local = os.path.join(save_dir_2,os.path.basename(segments[i]['Image File'][j]))
                image.save(save_local)
                save_local = os.path.join(save_dir_3, os.path.basename(segments[i]['Image File'][j]))
                scaled_object.save(save_local)

