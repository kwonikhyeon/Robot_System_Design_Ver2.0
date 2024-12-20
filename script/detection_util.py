# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.sensor import Camera

import random
import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO
import math
# ./python.sh -m pip install ultralytics

# 현재 스크립트 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))

# 상위 폴더 경로 가져오기
parent_dir = os.path.dirname(current_dir)

# Env
HOME = current_dir
width = 1920
height = 1080

# Load YOLO model
model = YOLO(f"{HOME}/best.pt")

def save_rgb_image(rgb, file_name):
    cv2.imwrite(file_name + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

def save_depth_image(depth, file_name, mode):
    # depth normalization
    min_depth, max_depth = depth.min(), depth.max()

    # mode 1: task 1
    if mode == 1:
        # 특정 height 만 보고 싶음
        depth[depth > min_depth + 0.001] = max_depth
    # mode 2: needle, do nothing

    # mode 3: hole_threshing
    elif mode == 3:
        # 특정 height 만 보고 싶음
        depth[depth > max_depth - 0.1] = max_depth

    depth = (depth - min_depth) / (max_depth - min_depth) * 255
    depth = depth.astype('uint8')
    
    cv2.imwrite(file_name + "_depth.png", cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))

def get_3dcoord(my_camera:Camera, target_point, depth):
    point_2d = np.array([target_point])
    # depth = np.array(depths[int(target_point[1]), int(target_point[0])])

    result_coord = my_camera.get_world_points_from_image_coords(point_2d, depth)
    return result_coord

def camera_init(my_camera:Camera):
    my_camera.initialize()

    my_camera.set_focal_length(1.93)
    my_camera.set_focus_distance(4)
    my_camera.set_horizontal_aperture(2.65)
    my_camera.set_vertical_aperture(1.48)

    my_camera.set_clipping_range(0.01, 10000)

    my_camera.add_distance_to_image_plane_to_frame()
    my_camera.add_distance_to_camera_to_frame()
    my_camera.add_instance_segmentation_to_frame()
    my_camera.add_pointcloud_to_frame()


def identify_shape(contour):
    shape = "unidentified"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        shape = "square" if 0.95 <= w / float(h) <= 1.05 else "rectangle"
    elif len(approx) == 6:
        shape = "hexagonal_prism"

        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        if (major_axis / minor_axis) > 1.2:
            shape = "needle"
    else:
        shape = "circle"
        
        ellipse = cv2.fitEllipse(contour)
        (center, axes, orientation) = ellipse
        major_axis, minor_axis = max(axes), min(axes)
        if (major_axis / minor_axis) > 1.2:
            shape = "needle"

    return shape


def find_parent_contours(contours, hierarchy):
    parent_contours = []
    donut_contours= []

    for idx, contour in enumerate(contours):
        idx1 = hierarchy[0][idx][0]
        idx2 = hierarchy[0][idx][1]
        idx3 = hierarchy[0][idx][2]
        
        # 최상위 parent나 child가 아니면
        if not (idx1 == -1 and idx2 == -1):
            child_shapes = []
            if idx3 == -1:
                parent_contours.append((idx, identify_shape(contour)))
            else:
                donut_contours.append((idx, identify_shape(contour), identify_shape(contours[idx3])))
    
    # Compare areas of the donut contours
    if len(donut_contours) == 2:

        area1 = cv2.contourArea(contours[donut_contours[0][0]]) - cv2.contourArea(contours[donut_contours[0][0] + 1])
        area2 = cv2.contourArea(contours[donut_contours[1][0]]) - cv2.contourArea(contours[donut_contours[1][0] + 1])

        if area1 > area2:
            parent_contours.append((donut_contours[0][0], "torus"))
            parent_contours.append((donut_contours[1][0], "tube"))
        else:
            parent_contours.append((donut_contours[1][0], "torus"))
            parent_contours.append((donut_contours[0][0], "tube"))

    return parent_contours

def getInfoObejct(my_camera:Camera, depths):
    img = cv2.imread(f"{HOME}/object_images/test_rgb.png")
    
    # Inference
    # Confidence 값 조정 필요 (hole과 franka 오판단 방지)
    # =================
    # RESULT
    # 0 cuboid
    # 1 cylinder
    # 2 hexagonal_prism
    # 3 needle
    # 4 torus
    # 5 tube    


    if os.name == 'nt':
        dest_folder = os.path.join(f"{parent_dir}\\script\\RESULT\\predict\\")
    else:
        dest_folder = os.path.join(f"{parent_dir}/RESULT/predict/")
    
    # 프로그램 매 실행마다 predict folder 제거
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
        print(f"removed {dest_folder}")
    else:
        print(f"{dest_folder} does not exits")

    results = model.predict(img, save=True, save_txt=True, save_conf=True, imgsz=640, conf=0.65, project='RESULT', name='predict')
    

    result_dict = {}
    if os.name == 'nt':
        img_dir = f"{parent_dir}\\script\\RESULT\\predict\\labels\\image0.txt"
    else:
        img_dir = f"{parent_dir}/RESULT/predict/labels/image0.txt"
    with open(img_dir, 'r') as file:
        for line in file:
            values = line.split()[:5]
            
            # local 좌표
            key = int(values[0])
            point_2d = np.array([(float(values[1]) * width, float(values[2]) * height)])
            # print(point_2d[0][1], point_2d[0][0])
            depth = np.array(depths[int(point_2d[0][1]), int(point_2d[0][0])])

            value = my_camera.get_world_points_from_image_coords(point_2d, depth)

            result_dict[key] = value
    
    # print(result_dict)
    
    # dict: {0: (x0, y0), 1: (x1, y1), ..., 5: (x5, y5)}
    return result_dict


def getInfoHole(my_camera:Camera, depths):
    img = cv2.imread(f'{HOME}/hole_images/test_depth.png', cv2.IMREAD_GRAYSCALE)

    # Convert Image
    img = 255 - img

    # Define mode
    mode = cv2.RETR_TREE  # Use RETR_TREE to get hierarchy information
    name = 'RETR_TREE'

    # Find contours
    contours, hierarchy = cv2.findContours(img, mode, cv2.CHAIN_APPROX_SIMPLE)
    depth = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # print(hierarchy)

    shape_dict = find_parent_contours(contours, hierarchy)
    # for idx, shape in shape_dict:
    #     print(f"Parent Contour {idx}: Shape={shape}")=

    # =================
    # RESULT
    # 0 cuboid
    # 1 cylinder
    # 2 hexagonal_prism
    # 3 needle
    # 4 torus
    # 5 tube   

    result_dict = {}

    # Process each parent contour
    for idx, shape in shape_dict:
        contour = contours[idx]
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate the corner points
        endpoints = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        
        # Draw the contour and bounding box
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # Random BGR value
        cv2.drawContours(depth, [contour], -1, c, 2, cv2.LINE_8)
        for point in endpoints:
            cv2.circle(depth, point, 5, (0, 0, 255), -1)  # Red circles for endpoints

        # Annotate the image with shape and contour index
        cv2.putText(depth, f"{shape} ({idx})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)


        # print([(x + w / 2, y + h / 2)])

        # 2d -> world coordinate
        point_2d = np.array([(x + w / 2, y + h / 2)])
        value = my_camera.get_world_points_from_image_coords(point_2d, depths.min())
        value[0][0] = round(value[0][0],2)
        value[0][1] = round(value[0][1],2)
        
        if shape == "rectangle":
            result_dict[0] = value
        elif shape == "circle":
            result_dict[1] = value
        elif shape == "hexagonal_prism":
            result_dict[2] = value
        elif shape == "needle":
            result_dict[3] = value
        elif shape == "torus":
            result_dict[4] = value
        elif shape == "tube":
            result_dict[5] = value

    cv2.imwrite(f"{HOME}/hole_images/test_depth_BB.png", cv2.cvtColor(depth, cv2.COLOR_RGB2BGR))
    
    return result_dict

# ============
# task 2

def detect_circles():
    img = cv2.imread(f'{HOME}/hole_images/test_depth_Needle.png', cv2.IMREAD_GRAYSCALE)
    img = cv2.medianBlur(img, 5)

    # Apply HoughCircles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1.27, minDist=30,
                               param1=100, param2=30, minRadius=10, maxRadius=80)

    # init
    mark_x = 0
    mark_y = 0

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            mark_x = x
            mark_y = y

            cv2.circle(img, (x, y), r, (0, 128, 255), 4)
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    
    cv2.imwrite(f"{HOME}/hole_images/test_depth_Needle_w_circle.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return mark_x, mark_y

def getInfoNeedle(my_camera:Camera, depths):
    # =================
    # RESULT
    # 3 needle
    # 4 needle mark

    result_dict = {}

    img = cv2.imread(f'{HOME}/hole_images/test_depth.png', cv2.IMREAD_GRAYSCALE)

    img = 255 - img
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)

    cv2.imwrite(f"{HOME}/hole_images/test_depth_Needle_pre.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding box for each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw the bounding box
    
    # 2d -> world coordinate
    point_2d = np.array([(x + w / 2, y + h / 2)])
    value = my_camera.get_world_points_from_image_coords(point_2d, depths.min())
    value[0][2] = 0
    result_dict[3] = value

    cv2.imwrite(f"{HOME}/hole_images/test_depth_Needle.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # needle mark coordinate
    x_, y_ = detect_circles()

    # 2d -> world coordinate
    point_2d = np.array([(x_, y_)])
    value = my_camera.get_world_points_from_image_coords(point_2d, depths.min())
    value[0][2] = 0
    result_dict[4] = value

    return result_dict

def getInfoHole_threading(my_camera:Camera, depths):
    # =================
    # RESULT
    # Hole_threading

    img = cv2.imread(f'{HOME}/hole_images/test_depth.png', cv2.IMREAD_GRAYSCALE)
    img = 255 - img
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw bounding box for each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw the bounding box
    
    # 2d -> world coordinate
    point_2d = np.array([(x + w / 2, y + h / 2)])
    value = my_camera.get_world_points_from_image_coords(point_2d, depths.min())
    value[0][2] = 0

    cv2.imwrite(f"{HOME}/hole_images/test_depth_BB.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return value[0]