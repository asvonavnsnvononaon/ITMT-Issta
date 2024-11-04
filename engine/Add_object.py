import cv2
import numpy as np
from PIL import Image
import random
from scipy import ndimage
def process_masks(mask, mask_obj):
    road_mask = mask == 0
    object_mask = mask_obj > 0
    h, w = road_mask.shape
    object_mask[3 * h // 4:, :] = 0
    ###########
    kernel_small = np.ones((3, 3), np.uint8)
    kernel_large = np.ones((5, 5), np.uint8)
    object_mask = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel_small)
    object_mask = cv2.dilate(object_mask, kernel_large, iterations=1) > 0
    #######
    non_zero_rows = np.where(np.any(object_mask, axis=1))[0]

    if len(non_zero_rows) > 0:
        valid_rows = non_zero_rows[:int(len(non_zero_rows) * 0.9)]
        last_row_index = valid_rows[-1]
        last_row = object_mask[last_row_index]
        non_zero_cols = np.nonzero(last_row)[0]
        left_20_percent = non_zero_cols[:max(1, int(len(non_zero_cols) * 0.2))]
        random_point = np.random.choice(left_20_percent)
        ###################
#        road_pixels_in_row = np.where(road_mask[last_row_index])[0]
#        road_width = road_pixels_in_row[-1] - road_pixels_in_row[0] + 1
        return (last_row_index, random_point)
    else:
        road_non_zero_rows = np.where(np.any(road_mask, axis=1))[0]
        valid_rows = road_non_zero_rows[int(len(road_non_zero_rows) * 0.4):int(len(road_non_zero_rows) * 0.6)]
        random_row = np.random.choice(valid_rows)
        road_pixels = np.where(road_mask[random_row])[0]
        rightmost_point = road_pixels[-1]
        return (random_row, rightmost_point)
from rembg import remove
def calculate_road_width(road_mask, point):
    row = point[0]
    road_pixels_in_row = np.where(road_mask[row] == 0)[0]
    if len(road_pixels_in_row) > 0:
        return road_pixels_in_row[-1] - road_pixels_in_row[0] + 1
    else:
        return 100
def get_mask_width(mask):
    non_zero = cv2.findNonZero(mask)
    x_coords = non_zero[:,:,0]
    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    width = max_x - min_x + 1
    return width


def integrated_road_width(mask, mask_obj, point):
    width = calculate_road_width(mask, point)
    width_1 = get_mask_width(mask_obj)
    lower_bound = width * 1/4
    upper_bound = width * 3/5
    if lower_bound <= width_1 <= upper_bound:
        return width_1
    else:
        return random.uniform(width * 1/3, width * 1/2)






def crop_image_and_mask_to_content(image, mask):
    mask_array = np.array(mask)
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    cropped_image = image.crop((xmin, ymin, xmax + 1, ymax + 1))
    cropped_mask_array = mask_array[ymin:ymax + 1, xmin:xmax + 1]
    cropped_mask = Image.fromarray(cropped_mask_array.astype(np.uint8))
    return cropped_image, cropped_mask


def generate_mask(image):
    object_mask = remove(image, only_mask=True)
    return crop_image_and_mask_to_content(image, object_mask)


def create_mask(image):
    img = image.convert("RGB")
    data = np.array(img)
    mask = (data[:,:,:3] != 255).any(axis=2)
    mask = ndimage.binary_dilation(mask)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_erosion(mask)
    return img,Image.fromarray((mask * 255).astype(np.uint8))



