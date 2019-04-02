from utils.utils import *
import os
import boxx
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import time
from tqdm import tqdm

# # Functions
def get_json_images_df(json_filepath):

    json = boxx.loadjson(json_filepath)

    json_images_df = pd.DataFrame()

    for i in tqdm(range(len(json['images']))):
        file_name = json['images'][i]['file_name']
        image_width = json['images'][i]['width']
        image_height = json['images'][i]['height']
        image_id = json['images'][i]['id']

        record = pd.DataFrame([[file_name, image_width, image_height, image_id]],
                              columns=['file_name', 'image_width', 'image_height', 'image_id'])

        json_images_df = json_images_df.append(record, ignore_index=True)

    return json_images_df


def get_json_annotations_df(json_filepath):

    json = boxx.loadjson(json_filepath)

    json_annotations_df = pd.DataFrame()

    for i in tqdm(range(len(json['annotations']))):
        image_id = json['annotations'][i]['image_id']
        image_area = json['annotations'][i]['area']
        image_bbox = json['annotations'][i]['bbox']
        image_point_xy = json['annotations'][i]['point_xy']
        image_segmentation = json['annotations'][i]['segmentation']
        is_crowded = json['annotations'][i]['iscrowd']
        category_id = json['annotations'][i]['category_id']

        record = pd.DataFrame([[image_id, image_area, image_bbox, image_point_xy,
                                image_segmentation, is_crowded, category_id]],
                              columns=['image_id','image_area','image_bbox','image_point_xy',
                                       'image_segmentation','is_crowded','category_id'])

        json_annotations_df = json_annotations_df.append(record, ignore_index=True)

    return json_annotations_df


def get_json_items_df(json_filepath):

    json = boxx.loadjson(json_filepath)

    json_items_df = pd.DataFrame()

    for i in tqdm(range(len(json['__raw_Chinese_name_df']))):
        sku_name = train_js['__raw_Chinese_name_df'][i]['sku_name']
        category_id = train_js['__raw_Chinese_name_df'][i]['category_id']
        sku_class = train_js['__raw_Chinese_name_df'][i]['sku_class']
        code = train_js['__raw_Chinese_name_df'][i]['code']
        shelf = train_js['__raw_Chinese_name_df'][i]['shelf']
        num = train_js['__raw_Chinese_name_df'][i]['num']
        name = train_js['__raw_Chinese_name_df'][i]['name']
        clas = train_js['__raw_Chinese_name_df'][i]['clas']
        known = train_js['__raw_Chinese_name_df'][i]['known']
        ind = train_js['__raw_Chinese_name_df'][i]['ind']

        record = pd.DataFrame([[sku_name, category_id, sku_class, code, shelf, num, name, clas, known, ind]],
                              columns=['sku_name','category_id','sku_class','code','shelf','num','name','clas','known','ind'])
        json_items_df = json_items_df.append(record, ignore_index=True)

    return json_items_df


def get_json_df(json_filepath, data_dir):

    json_images_df = get_json_images_df(json_filepath)
    json_annotations_df = get_json_annotations_df(json_filepath)

    json_df = json_images_df.merge(json_annotations_df,
                                   how='left',
                                   left_on='image_id',
                                   right_on='image_id')

    json_df.insert(0, 'data_directory', data_dir)

    return json_df


def get_yolo_annotations(image_width, image_height, image_bbox):

    x_min = image_bbox[0]
    y_min = image_bbox[1]
    x_max = image_bbox[0] + image_bbox[2]
    y_max = image_bbox[1] + image_bbox[3]
    x = (x_min + x_max) / 2.0 / image_width
    y = (y_min + y_max) / 2.0 / image_height
    w = (x_max - x_min) / image_width
    h = (y_max - y_min) / image_height

    yolo_bbox = [x, y, w, h]

    return yolo_bbox


def from_yolo_to_cor(box, shape):
    img_h, img_w, _ = shape
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    return x1, y1, x2, y2


def draw_boxes(img, boxes, shape):
    for box in boxes:
        x1, y1, x2, y2 = from_yolo_to_cor(box, shape)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
    plt.imshow(img)


train_js = boxx.loadjson('/data/instances_train2019.json')

train_json_filepath = '/data/instances_train2019.json'
json_items_df = get_json_items_df(train_json_filepath)
train_json_df = get_json_df(train_json_filepath, 'data/train2019/')
train_json_df['yolo_bbox'] = train_json_df[['image_width','image_height','image_bbox']].apply(lambda x: get_yolo_annotations(x[0],x[1],x[2]),axis=1)

val_json_filepath = '/data/instances_val2019.json'
val_json_df = get_json_df(val_json_filepath, 'data/val2019/')
val_json_df['yolo_bbox'] = val_json_df[['image_width','image_height','image_bbox']].apply(lambda x: get_yolo_annotations(x[0],x[1],x[2]),axis=1)

test_json_filepath = '/data/instances_test2019.json'
test_json_df = get_json_df(test_json_filepath, 'data/test2019/')
test_json_df['yolo_bbox'] = test_json_df[['image_width','image_height','image_bbox']].apply(lambda x: get_yolo_annotations(x[0],x[1],x[2]),axis=1)

# # Select particular image and display bounding box
i = 612
img = np.array(Image.open(os.path.join('data/val2019/',val_json_df['file_name'].values[i])))

draw_boxes(img, list(val_json_df['yolo_bbox'][val_json_df['image_id']==val_json_df['image_id'].values[i]]),
           (val_json_df['image_height'].values[i],
            val_json_df['image_width'].values[i],
            3))
