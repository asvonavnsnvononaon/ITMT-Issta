import torch
import sys
import os
from openpyxl import Workbook
#from Script.remove_anything_video import remove_objects_from_video,process_objects
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import Script.Diffusion_tools as diffusion_utils
import Script.Autonomous_tools as autonomous_tools
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import json
#import Script.diffusion
from scipy.interpolate import splprep, splev
import cv2
import os
from rembg import remove
import Script.auto_models
import pandas as pd
import random
import engine.data_process as data_process
from segment_anything import sam_model_registry, SamPredictor
from scipy import interpolate
import Script.diffusion as diffusion
import random
import shutil
from scipy.ndimage import distance_transform_edt
from engine.RIFE import interpolate

import PIL
import requests
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import matplotlib.pyplot as plt
def get_informations(args):
    if args.dataset == "udacity":
        datasets = ["HMB1", "HMB2", "HMB4", "HMB5", "HMB6"]
    else:
        datasets = ["camera_lidar-20180810150607", "camera_lidar-20190401121727", "camera_lidar-20190401145936"]
    dir_1 = os.path.join(args.data_file, "ADS_data", "torch", args.dataset)#改变大小的图片的位置
    dir_2 = os.path.join(args.data_file, "ADS_data", "torch", args.dataset, "OneFormer")#分割图像的位置，尺寸是对应原来的图像
    return dir_1,dir_2,datasets




def MT(args,MRs):
    if args.MT_process==False:
        return 0
    args.new_test_data=False
    if args.new_test_data==True:
        find_test_images(args)

    statistic_object =["Traffic sign","traffic signals","other road infrastructure"]
    replace_object = ["road markings","weather elements"]
    equality_MRs = ["equality MRs"]
    dynamics_object =["traffic participants"]
    #MRs["class"] = "weather elements"
    if MRs["class"] in replace_object:
        MT_replace_object(args)

    #if MRs["class"] in statistic_object:
    #    MT_statistic_object(args)
    if MRs["class"] in dynamics_object:
        pass




        if MRs['class_1']=="person":
            MT_dynamics_object(args,MRs)
        else:
            MT_statistic_object(args, MRs)

def MT_replace_object(args,order="small white stop line"):
    dir_1, dir_2, datasets = get_informations(args)
    with open(os.path.join(args.data_file, "MT_results","info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    processed_images = []
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                                                  safety_checker=None)
    pipe.to("cuda")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    save_dir_1 = os.path.join(args.data_file, "MT_results", "follow_up")
    for idx in range(0, len(data_lists), 5):
        data_list = data_lists[idx:idx + 5]
        road_masks = []
        original_images = []


        for count, item in enumerate(data_list, start=0):
            if count%2==0:
                image = Image.open(item['Image File'])
                prompt = f"what would it look like if it were {order}"
                images = pipe(prompt, image=image, guidance_scale=10, num_inference_steps=10,
                              image_guidance_scale=1.5).images
                save_local = os.path.join(save_dir_1, os.path.basename(item['Image File']))
                images[0].save(save_local)
                processed_images.append(save_local)
            else:
                processed_images.append(item['Image File'])
            if count >= 2 and count % 2 == 0:
                if count >= 2 and count % 2 == 0:
                    fig_new = interpolate(processed_images[-3],processed_images[-1])
                    save_local = os.path.join(save_dir_1, os.path.basename(processed_images[-2]))
                    cv2.imwrite(save_local,fig_new)





def find_test_images(args):
    dir_1, dir_2, datasets = get_informations(args)
    random_dataset = random.choice(datasets)
    save_dir = os.path.join(args.data_file, "MT_results","original")
    save_dir_1 = os.path.join(args.data_file, "MT_results", "follow_up")
    save_dir_2 = os.path.join(args.data_file, "MT_results", "mask")
    for dir in [save_dir, save_dir_1, save_dir_2]:
        shutil.rmtree(dir, ignore_errors=True)
        data_process.Check_file(dir)
    ######################################################################################
    rs = pd.read_excel(os.path.join(dir_1, random_dataset, "matched_data.xlsx"), header=0)
    rs = rs.iloc[1:-1]
    start_index = random.randint(0, 1000)
    time_series = 5
    number_test_images = args.MT_image_num-args.MT_image_num%time_series
    if args.dataset =="A2D2":
        Turn_threshold_low = 0.2# 同样大于0是左转 不过A2D2不是归一化的数据是真是转弯角度 所以有更大的判断角度
        Turn_threshold_upper = 0.9  #
    else:
        Turn_threshold_low = 0.04 #转弯的角度大于0是向左转 小于0是向右转
        Turn_threshold_upper = 0.15  # 转弯的角度大于0是向左转 小于0是向右转
    Saved_images_number = 0
    data_list = []
    #########################################选择5个图片
    while Saved_images_number<number_test_images:
        if start_index + time_series<len(rs["Timestamp"]):
            selected = rs[start_index:start_index + time_series]
        else:
            random_dataset = random.choice(datasets)
            rs = pd.read_excel(os.path.join(dir_1, random_dataset, "matched_data.xlsx"), header=0)
            rs = rs.iloc[1:-1]
            start_index = random.randint(0, 1000)
            selected = rs[start_index:start_index + time_series]
        selected = selected.reset_index(drop=True)
        start_index = start_index + time_series
        speeds = np.mean(selected['Vehicle Speed'])
        steering_angles = []
        ########################################
        # 询问LLM对于速度和方向盘方向
        #######################################

        avg_angle = np.mean(selected['Steering Angle'])
        avg_speed = np.mean(selected['Vehicle Speed'])

        if args.direction =="forward":
            if abs(avg_angle)>Turn_threshold_low:
                continue
        elif args.direction =="Turn left":
            if avg_angle<Turn_threshold_upper or abs(avg_speed)<3:
                continue
        elif args.direction == "Turn right":
            if avg_angle > -Turn_threshold_upper or abs(avg_speed)<3:
                continue

        for idx in range(time_series):
            image = os.path.basename(selected['Image File'][idx])
            parts = selected['Image File'][idx].split(os.sep)
            index = parts.index(random_dataset)
            parts.insert(index, 'OneFormer')
            new_path = os.sep.join(parts)
            mask = Image.open(new_path)
            save_path_1 = os.path.join(save_dir_2, image)
            mask.save(save_path_1)
            img = Image.open(selected['Image File'][idx])
            save_path = os.path.join(save_dir, image)
            save_path_2 = os.path.join(save_dir_1, image)
            img.save(save_path)
            data = {
                'Image File': save_path,
                'Mask File': save_path_1,
                'result File': save_path_2,
                'Steering Angle': selected['Steering Angle'][idx],
                'Vehicle Speed': selected['Vehicle Speed'][idx],
            }

            data_list.append(data)
        Saved_images_number = Saved_images_number+5
    ####################################################################################
    with open(os.path.join(args.data_file, "MT_results", "info.json"), 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
    # 创建Excel工作簿和工作表
    wb = Workbook()
    ws = wb.active
    ws.title = "Matched Data"
    # 添加表头
    headers = ["Image File", "Mask File", 'result File', "Steering Angle", "Vehicle Speed"]
    ws.append(headers)

    # 添加匹配的数据
    for data in data_list:
        ws.append([data['Image File'], data['Mask File'], data['result File'], data['Steering Angle'],
                   data['Vehicle Speed']])

    # 保存Excel文件
    excel_filename = os.path.join(args.data_file, "MT_results", "matched_data.xlsx")
    wb.save(excel_filename)


################################################
def MT_statistic_object(args,MRs,scale=2):
    dir_1, dir_2, datasets = get_informations(args)
    white_bbox = find_reference_point(args,MRs)
    with open(os.path.join(args.data_file, "MT_results","info.json"), 'r', encoding='utf-8') as file:
        data_lists = json.load(file)
    idx = 0
    save_dir_1 = os.path.join(args.data_file, "MT_results", "follow_up")
    shutil.rmtree(save_dir_1, ignore_errors=True)
    data_process.Check_file(save_dir_1)
    reference_images,reference_masks = Get_reference_images(args,MRs)
    length = len(reference_images)
    processed_images = []
    for idx in range(0, len(data_lists), 5):
        data_list = data_lists[idx:idx + 5]
        road_masks = []
        original_images = []

        for item in data_list:
            road_mask = cv2.imread(item['Mask File'], cv2.IMREAD_GRAYSCALE)
            road_masks.append(road_mask)
            original_images.append(cv2.imread(item['Image File']))

        best_mask = select_narrowest_road_mask(road_masks)
        ###############################################
        #计算道路最近点的坐标
        bbox = white_bbox[8]

        suitable_x, suitable_y = search_nearest_road_point(best_mask, bbox)
        dx = suitable_x - bbox[0]  # x方向的移动距离
        dy = suitable_y - bbox[3]  # y方向的移动距离（注意bbox[3]是底部y坐标）

        for item, image in zip(data_list, original_images):
            overlay = image.copy()
            road_color = [0, 255, 0]  # 绿色表示道路
            overlay[best_mask == 6] = road_color

            # 混合原始图像和叠加层
            alpha = 0.4  # 透明度
            result = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

            # 获取并绘制边界框
            bbox_color = (255, 0, 0)  # 蓝色
            cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)

            # 查找合适的位置并绘制
            #suitable_x, suitable_y = find_suitable_position_with_bbox(best_mask, bbox, MRs)
            if suitable_x is not None and suitable_y is not None:
                cv2.circle(result, (suitable_x, suitable_y), 5, (0, 0, 255), -1)  # 红色圆点

            # 保存结果
            save_local = os.path.join(save_dir_1, os.path.basename(item['Image File']))
            cv2.imwrite(save_local, result)


        """
        MRs["location"] = "on the roadside"
        if MRs["location"] == "on the roadside":
            for count, item in enumerate(data_list, start=0):
                original_bbox = white_bbox[8]
                bbox = (original_bbox[0] + dx, original_bbox[1] + dy,  original_bbox[2] + dx, original_bbox[3] + dy )
                bbox_width = bbox[2] - bbox[0]
                reference_image = reference_images[idx % length]
                reference_mask = reference_masks[idx % length]
                original_width, original_height = reference_image.size

                scale_factor = bbox_width / original_width * scale
                new_height = int(original_height * scale_factor)
                new_width = int(bbox_width * scale)
                resized_reference_image = reference_image.resize((new_width, new_height), Image.LANCZOS)
                resized_reference_mask = reference_mask.resize((new_width, new_height), Image.LANCZOS)
                if count%2==0:
                    image = Image.open(item['Image File'])
                    save_local = os.path.join(save_dir_1, os.path.basename(item['Image File']))
                    image.paste(resized_reference_image, ( bbox[0], bbox[3]-new_height), mask=resized_reference_mask)
                    image.save(save_local)
                    processed_images.append(save_local)
                else:
                    processed_images.append(item['Image File'])
                if count >= 2 and count % 2 == 0:
                    if count >= 2 and count % 2 == 0:
                        fig_new = interpolate(processed_images[-3],processed_images[-1])
                        save_local = os.path.join(save_dir_1, os.path.basename(processed_images[-2]))
                        cv2.imwrite(save_local,fig_new)

        if MRs["location"] == "along the same road":
            bbox = [110,100,170,140]
            result = cv2.imread(item['Image File'])
            save_local = os.path.join(save_dir_1, os.path.basename(item['Image File']))


            bbox_color = (255, 0, 0)  # 蓝色
            cv2.rectangle(result, (bbox[0], bbox[1]), (bbox[2], bbox[3]), bbox_color, 2)
            cv2.imwrite(save_local, result)

        if MRs["location"] == "along the same road":
            continue
            road_pixels = best_mask == 6

            # 计算每行的道路宽度
            road_widths = np.sum(road_pixels, axis=1)

            def find_central_connected_block(best_mask):
                # 创建一个二值掩码，其中20的像素为1，其他为0
                best_mask = best_mask[:, :, 0]
                binary_mask = (best_mask == 20).astype(np.uint8)

                # 使用连通组件分析找出所有连续的像素块
                num_labels, labels = cv2.connectedComponents(binary_mask)

                if num_labels == 1:  # 只有背景，没有找到任何块
                    return None

                # 计算掩码的中心点
                center_y, center_x = np.array(best_mask.shape) // 2

                # 初始化变量以存储最中心的块
                min_distance = float('inf')
                central_block = None

                for label in range(1, num_labels):  # 跳过背景（标签0）
                    # 获取当前块的坐标
                    y_coords, x_coords = np.where(labels == label)

                    # 计算块的中心
                    block_center_y = np.mean(y_coords)
                    block_center_x = np.mean(x_coords)

                    # 计算块中心到掩码中心的距离
                    distance = np.sqrt((block_center_y - center_y) ** 2 + (block_center_x - center_x) ** 2)

                    # 如果这个块更靠近中心，更新最中心的块
                    if distance < min_distance:
                        min_distance = distance
                        central_block = (labels == label).astype(np.uint8)

                if central_block is not None:
                    # 创建白色背景黑色前景的掩码
                    final_mask = 255 * np.ones_like(central_block, dtype=np.uint8)
                    final_mask[central_block == 1] = 0
                    return final_mask
                else:
                    return None
            for count, item in enumerate(data_list, start=0):
                original_bbox = white_bbox[8]
                bbox = [1, 60, 200, 140]
                mask_new = cv2.imread(item['Mask File'])
                save_local = os.path.join(save_dir_1, os.path.basename(item['Image File']))
                final_mask = find_central_connected_block(mask_new)
                if final_mask!=None:
                    cv2.imwrite(save_local, final_mask)



        for item in data_list:
            continue
            bbox = white_bbox[item+20]
#            result = results[idx_reference]
            bbox_width = bbox[2] - bbox[0]
            reference_image = reference_images[idx % length]
            reference_mask = reference_masks[idx % length]
            original_width, original_height = reference_image.size
            scale_factor = bbox_width / original_width*scale
            new_height = int(original_height * scale_factor)
            new_width = int(bbox_width*bbox_width)
            resized_reference_image = reference_image.resize((new_width, new_height), Image.LANCZOS)
            resized_reference_mask = reference_mask.resize((new_width, new_height), Image.LANCZOS)

            image = Image.open(item['Image File'])

            save_local = os.path.join(save_dir_1, os.path.basename(item['Image File']))
            road_mask = cv2.imread(item['Mask File'], cv2.IMREAD_GRAYSCALE)



            if MRs["location"]=="on the roadside":
                while point_x > 0:
                    road_area = road_mask[point_y:point_y + new_height, point_x:point_x + bbox_width]
                    # if not np.any(road_area == 6):  # 假设6表示道路
                    if not np.any((road_area == 6) | (road_area == 20) | (road_area == 102)):
                        break
                    point_x += 1
                point_x = min(point_x, 320 - bbox_width - 20)
                while point_y + new_height < 160:
                    if road_mask[point_y + new_height - 10, point_x] != 6 and road_mask[
                        point_y + new_height - 10, point_x] != 20:
                        point_y = point_y + 1
                    else:
                        break
            if MRs["location"] == "along the same road":

                while point_x > 0:
                    road_area = road_mask[point_y:point_y + new_height, point_x:point_x + bbox_width]
                    # if not np.any(road_area == 6):  # 假设6表示道路
                    if not np.any((road_area == 6)):
                        break
                    point_x += 1
                point_x = point_x - 80
                point_x = min(point_x, 320 - bbox_width - 20)
                while point_y + new_height < 160:
                    if road_mask[point_y + new_height - 5, point_x] != 6:
                        point_y = point_y + 1
                    else:
                        break
            if MRs["location"] == "crossing the road":
                while point_y + new_height < 160:
                    if road_mask[point_y + new_height - 5, point_x] != 6:
                        point_y = point_y + 1
                    else:
                        break
                if jj == 0:
                    past_point_x = point_x
                    past_point_y = point_y
                else:
                    if abs(point_x - past_point_x) > 20 or abs(point_y - past_point_y) > 20:
                        point_x = past_point_x
                        point_y = past_point_y
                remove_x = list(range(0, 20 * 5, 5))
                point_x = point_x-remove_x[idx_reference]

                jj = jj + 1
            if jj==0:
                past_point_x =point_x
                past_point_y = point_y
            else:
                if abs(point_x-past_point_x)>10 or abs(point_y-past_point_y)>10:
                    point_x = past_point_x
                    point_y = past_point_y
            jj=jj+1

            image.paste(resized_reference_image, ( point_x, point_y), mask=resized_reference_mask)


            image.save(save_local)
            """

#scale_bbox_1. 先保存一个背景白色的，然后构成一个mask，然后计算出被遮挡的部分
#2.要添加的照片先保存在一个临时的文件夹（首先缩放插入之前的大小），用参数决定是否操作


def find_rightmost_edge_polygon(data_list):
    all_right_edges = []
    for item in data_list:
        road_mask = cv2.imread(item['Mask File'], cv2.IMREAD_GRAYSCALE)
        road_mask = np.where(road_mask == 6, 255, 0).astype(np.uint8)
        # 找出每行最右边的点
        right_edge = []
        height, width = road_mask.shape
        for y in range(height):
            row = road_mask[y]
            right_points = np.where(row == 255)[0]
            if right_points.size > 0:
                right_edge.append((right_points[-1], y))

        if right_edge:
            all_right_edges.append(right_edge)

    if not all_right_edges:
        return None
        # 找出最靠左的右边缘
    leftmost_right_edge = min(all_right_edges, key=lambda edge: min(point[0] for point in edge))

    # 按y坐标排序
    leftmost_right_edge.sort(key=lambda point: point[1])

    # 选择20个均匀分布的点
    total_points = len(leftmost_right_edge)
    if total_points >= 20:
        indices = np.linspace(0, total_points - 1, 20, dtype=int)
        selected_points = [leftmost_right_edge[i] for i in indices]
    else:
        # 如果点数少于20，通过线性插值生成20个点
        x = [p[0] for p in leftmost_right_edge]
        y = [p[1] for p in leftmost_right_edge]
        new_y = np.linspace(min(y), max(y), 20)
        new_x = np.interp(new_y, y, x)
        selected_points = list(zip(new_x.astype(int), new_y.astype(int)))

    return selected_points

def Get_reference_images(args,MRs):
    images = []
    masks = []
    statistic_object = ["Traffic sign", "traffic signals", "other road infrastructure"]
    replace_object = ["road markings", "weather elements"]
    equality_MRs = ["equality MRs"]
    dynamics_object = ["traffic participants"]
    MRs["class"] = "Traffic sign"

    if MRs["class"] in statistic_object:
        MR_example = ["red light"]
        reference_image_path= "Data//MRs//Traffic sign//image//"+MR_example[0]+".png"
    reference_image = Image.open(reference_image_path)
    images.append(reference_image)
    reference_mask = create_mask(reference_image)
    masks.append(reference_mask)
    return images,masks

def find_reference_point(args,MRs):
    location_example = ["on a different road","on the roadside","along the same road"]
    if MRs["location"] in location_example:
        if args.direction =="forward":
            with open(os.path.join(args.data_file, "Diffusion", "Reference", "A2D2", "scenario_9","results", "info.json"), 'r') as file:
                locations = json.load(file)
        else:
            with open(os.path.join(args.data_file, "Diffusion", "Reference", "A2D2", "scenario_5","results", "info.json"), 'r') as file:
                locations = json.load(file)
        white_bbox = []
        for idx in range(10):
            scaled_bbox = scale_bbox(locations[idx + 10]['white_bbox'])
            white_bbox.append(scaled_bbox)
    else:
        with open(os.path.join(args.data_file, "Diffusion", "Reference", "A2D2", "scenario_3", "results", "info.json"),'r') as file:
            locations = json.load(file)
        white_bbox = []
        for idx in range(10):
            scaled_bbox = scale_bbox(locations[idx + 20]['white_bbox'])
            white_bbox.append(scaled_bbox)
    return white_bbox

def scale_bbox(bbox, original_size=(1800, 1200), new_size=(320, 160)):
    orig_width, orig_height = original_size
    new_width, new_height = new_size

    x_scale = new_width / orig_width
    y_scale = new_height / orig_height

    x1, y1, x2, y2 = bbox

    new_x1 = int(x1 * x_scale)
    new_y1 = int(y1 * y_scale)
    new_x2 = int(x2 * x_scale)
    new_y2 = int(y2 * y_scale)

    return [new_x1, new_y1, new_x2, new_y2]



def analyze_MRs(MRs):
    with open(os.path.join("..", 'LLM_MT', 'Texas.json'), 'r', encoding='utf-8') as file:
        data_all = json.load(file)
    for item in data_all:
        pass











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

def MT_dynamics_object(args,MRs):
    pass












def Select_roadside_images(selected,save_dir):
    Reference_move_distance = 23
    Total = len(selected["Vehicle Speed"])
    idx = 0
    segments = []
    while idx<Total-45:
        distance = 0
        count = 20
        while distance<Reference_move_distance:
            distance = 0
            angle = 0
            for i in range(count):
                speed_ms = selected["Vehicle Speed"].iloc[idx + i]
                distance += speed_ms / 10
                angle =(angle+abs(selected["Steering Angle"].iloc[idx + i]))/2


            if distance < Reference_move_distance:
                count += 10  # 如果距离不足，增加1秒（10个0.1秒间隔）

            if count > 40:  # 限制最大间隔为5
                break

        if angle < 0.1:
            segment = {
                "Image File": selected["Image File"].iloc[idx:idx + count].tolist(), # 转换回秒
                "Steering Angle": selected["Steering Angle"].iloc[idx:idx + count].tolist(),
                "Vehicle Speed": selected["Vehicle Speed"].iloc[idx:idx + count].tolist()
            }
            segments.append(segment)
            idx += count
        else:
            idx += 1
        with open( os.path.join(save_dir, "speed_segments.json"), 'w') as f:
            json.dump(segments, f)
    return 0


def uniform_sample_dicts(data, num_samples, key='white_bbox'):
    if not data:
        return []

    # 如果数据量足够，进行均匀采样
    if num_samples <= len(data):
        interval = (len(data) - 1) / (num_samples - 1)
        indices = np.round(np.linspace(0, len(data) - 1, num_samples)).astype(int)
        return [data[i][key] for i in indices]
    else:
        # 对可用数据进行均匀采样
        sampled = [item[key] for item in data]
        # 找到最大的bbox
        max_bbox = max(sampled, key=lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        # 用最大的bbox填充剩余部分
        sampled.extend([max_bbox] * (num_samples - len(data)))
        return sampled

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
            size =1
            scale_factor = bbox_width / original_width  *size
            new_height = int(original_height * scale_factor*0.9*size)
            new_width = int(bbox_width*size)
            resized_reference_image = reference_image.resize((new_width, new_height), Image.LANCZOS)
            resized_reference_mask = reference_mask.resize((new_width, new_height), Image.LANCZOS)

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


def find_rightmost_edge_polygon_1(road_mask_path, num_points=100):
    # 读取图像
    # 读取mask
    road_mask = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)

    # 创建一个二值化mask，其中道路（值为6）变为255，其他区域为0
    road_mask = np.where(road_mask == 6, 255, 0).astype(np.uint8)

    # 找到轮廓
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 假设最大的轮廓是道路
    road_contour = max(contours, key=cv2.contourArea)

    # 使用多边形逼近简化轮廓
    epsilon = 0.02 * cv2.arcLength(road_contour, True)
    approx_polygon = cv2.approxPolyDP(road_contour, epsilon, True)

    # 找到多边形的最右边的边
    rightmost_edge = []
    for i in range(len(approx_polygon)):
        pt1 = approx_polygon[i][0]
        pt2 = approx_polygon[(i + 1) % len(approx_polygon)][0]
        if pt1[0] > road_mask.shape[1] / 2 and pt2[0] > road_mask.shape[1] / 2:  # 只考虑右半部分
            rightmost_edge.append((pt1[0], pt1[1]))
            rightmost_edge.append((pt2[0], pt2[1]))

    # 按y坐标排序
    rightmost_edge.sort(key=lambda x: x[1])

    # 插值
    if len(rightmost_edge) >= 2:
        # 将结果点转换为numpy数组
        points = np.array(rightmost_edge)

        # 计算每个点的累积距离
        distances = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
        distances = np.insert(distances, 0, 0)  # 在开始处插入0

        # 创建均匀分布的距离点
        equidistant_points = np.linspace(0, distances[-1], num_points)

        # 对x和y坐标分别进行插值
        interpolated_x = np.interp(equidistant_points, distances, points[:, 0])
        interpolated_y = np.interp(equidistant_points, distances, points[:, 1])

        # 将插值结果组合成点列表
        interpolated_result = list(zip(interpolated_x.astype(int), interpolated_y.astype(int)))

        return interpolated_result
    else:
        return rightmost_edge  # 如果点数少于2，返回原始结果

def scale_bbox_1(bbox, width_scale=2.5, height_scale=3):
    x_min, y_min, x_max, y_max = bbox
    new_x_min = int(x_min / width_scale)
    new_y_min = int(y_min / height_scale)
    new_x_max = int(x_max / width_scale)
    new_y_max = int(y_max / height_scale)
    return [new_x_min, new_y_min, new_x_max, new_y_max]


def find_rightmost_point(road_mask_path, reference_point):
    # 读取road_mask
    road_mask = cv2.imread(road_mask_path, cv2.IMREAD_GRAYSCALE)
    # 创建一个二值化mask，其中道路（值为6）变为255，其他区域为0
    road_mask = np.where(road_mask == 6, 255, 0).astype(np.uint8)

    if road_mask is None:
        print(f"Failed to read road mask: {road_mask_path}")
        return None

    # 找到所有连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(road_mask, connectivity=8)

    # 排除背景（通常是第一个组件）
    sizes = stats[1:, -1]
    num_labels = num_labels - 1

    # 找到最大的连通区域（假设为道路）
    max_label = np.argmax(sizes) + 1

    # 创建一个只包含最大连通区域的掩码
    road_mask = np.zeros(road_mask.shape, dtype=np.uint8)
    road_mask[labels == max_label] = 255

    # 确保reference_point的y在有效范围内
    row = reference_point[1]
    if row < 0 or row >= road_mask.shape[0]:
        print(f"Invalid row: {row}")
        return None

    # 获取指定行
    mask_row = road_mask[row]

    # 从右向左查找第一个非零点
    for x in range(mask_row.shape[0] - 1, -1, -1):
        if mask_row[x] > 0:
            return (x, row)

    print(f"No road edge found in row {row}")
    return None


from scipy import ndimage
def create_mask(image):
    # 转换图片为RGB模式
    img = image.convert("RGB")
    data = np.array(img)

    # 创建非白色像素的mask
    mask = (data[:,:,:3] != 255).any(axis=2)

    # 使用膨胀和腐蚀操作来填充内部的小白色区域
    mask = ndimage.binary_dilation(mask)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask)

    # 转换mask为图像
    return Image.fromarray((mask * 255).astype(np.uint8))


def MT_diffusion(args):
    dir_1, dir_2, dir_3, datasets = get_informations(args)

    reference = os.path.join(args.data_file, "Diffusion", "Reference", "A2D2", "scenario_9")
    selected_items = Select_Trafic_sign(args)
    ###########################################################################################
    selected_datasets = [datasets[1]]
    for selected_dataset in selected_datasets:
        save_dir = os.path.join(args.data_file, "MT_results", "Traffic sign",selected_dataset)
        seg_dir = os.path.join(dir_3,selected_dataset)

        data_process.Check_file(save_dir)

        rs = pd.read_excel(os.path.join(dir_1, selected_dataset, "matched_data.xlsx"), header=0)
        rs = rs.iloc[1:-1]
        if len(rs) <= args.MT_image_num:
            selected = rs
        else:
            #start_index = random.randint(0, len(rs) - 1000)
            start_index = 1000
            selected = rs[start_index:start_index+args.MT_image_num]
        #0   Timestamp      Image File      Steering Angle   Vehicle Speed
        # 读取 xlsx 文件
        Select_roadside_images(selected, save_dir)
        for num in range(len(selected_items)):
            MR_path = correct_path(selected_items[num]['Image Path'])
            reference_image_path = os.path.join(args.data_file, "MRs",MR_path)
            save_dir_1 = os.path.join(save_dir,str(num))
            data_process.Check_file(save_dir_1)
            Create_road_images(save_dir,save_dir_1,reference_image_path,reference,seg_dir)
            #print(image_path,MRs)

def Create_road_images(save_dir,save_dir_1,reference_image_path,reference,seg_dir):
    with open(os.path.join(reference,"results","info.json"), 'r') as file:
        locations = json.load(file)
    with open(os.path.join(save_dir,"speed_segments.json"), 'r') as f:
        segments = json.load(f)
    reference_image = Image.open(reference_image_path)
    reference_mask = create_mask(reference_image)
    prompt = "ambulance with red emergency light, viewed from behind, solo, isolated, no background, minimalist, centered."
    diffusion_type =  "mobius"
    reference_image = diffusion.Get_diffusion_image(prompt,diffusion_type)
    reference_mask = remove(reference_image, only_mask=True)

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
                point_x = rightmost_points[0]-bbox_width
                point_y = rightmost_points[1]-new_height//2
                past_point_x =point_x
                past_point_y = point_y
            else:
                point_x = rightmost_points[0]-bbox_width
                point_y = rightmost_points[1] - new_height//2
                if abs(point_x-past_point_x)>50:
                    point_x = past_point_x
                    point_y = past_point_y
            image.paste(resized_reference_image, (point_x, point_y), mask=resized_reference_mask)
            if point_x>image.width-bbox_width//2 or point_y<0:
                pass
            else:
                save_local = os.path.join(save_dir_2,os.path.basename(segments[i]['Image File'][j]))
                image.save(save_local)
                save_local = os.path.join(save_dir_3, os.path.basename(segments[i]['Image File'][j]))
                scaled_object.save(save_local)

def SAM_prediction_picture(image_rgb, clicked_points,input_labels=None, model_type="vit_h", checkpoint_path="models//sam_vit_h_4b8939-001.pth", device="cuda"):
    """执行预测并返回掩码、分数和逻辑值。"""
    model = sam_model_registry[model_type](checkpoint=checkpoint_path)
    model.to(device=device)
    predictor = SamPredictor(model)
    predictor.set_image(image_rgb)
    input_points = np.array(clicked_points)
    if input_labels is None:
        input_labels = np.ones(len(input_points), dtype=int)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )
    #masks = find_major(masks)

    return masks, scores, logits