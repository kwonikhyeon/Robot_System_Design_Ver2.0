import typing

import os
import numpy as np
import cv2
import math

from scipy.spatial.transform import Rotation as R
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.rotations import quat_to_euler_angles
import omni.isaac.core.utils.numpy.rotations as rot_utils

def find_closest_axis_to_world_z(angles):
    # 월드 좌표계의 z축 단위 벡터
    rotation = R.from_euler('XYZ', angles).as_matrix()
    world_z = np.array([0, 0, 1])
    
    # 각 로컬 축과 z축의 내적 계산
    similarities = [np.dot(axis, world_z) for axis in rotation]
    
    # 최대값의 인덱스 및 값 반환
    closest_axis_index = np.argmax(np.abs(similarities)) 
    axis = {0:"X-axis", 1:"Y-axis", 2:"Z-axis"}
    return axis[closest_axis_index], similarities[closest_axis_index]

def is_object_standing(object_name:str, closest_axis:str):
    if object_name == "cuboid" and closest_axis == "Z-axis":
        return True
    elif object_name == "cylinder" and closest_axis == "Z-axis":
        return True
    elif object_name == "hexagonal_prism" and closest_axis == "Z-axis":
        return True
    elif object_name == "needle" and closest_axis != "Z-axis":
        return True
    elif object_name == "torus" and closest_axis != "Z-axis":
        return True
    elif object_name == "tube" and closest_axis == "Z-axis":
        return True
    elif object_name == "needle_threading" and closest_axis != "Z-axis":
        return True
    return False

def get_offset_z_position(object_name:str, position:float, is_standing:bool):
    # {"cuboid": 0, "cylinder": 1, "hexagonal_prism": 2, "needle": 3, "torus": 4, "tube": 5 }
    offset_z = position    

    if object_name == "cuboid":
        offset_z = position * 1.5  
    elif object_name == "cylinder":
        pass
    elif object_name == "hexagonal_prism":
        pass
    
    elif object_name == "needle":
        if is_standing:
            offset_z = position / 2.0            
        else:
            offset_z = position * 2.0            
    
    elif object_name == "torus":
        offset_z = position * 2.0
    elif object_name == "tube":
        pass
    else:
        pass

    return offset_z

def compute_new_coordinate(position, angle, r):
    """
    현재 위치 (x, y)에서 주어진 각도(angle, radian)와 거리(r)를 이용하여
    새로운 좌표 (x_new, y_new)를 계산하는 함수입니다.

    Args:
        x (float): 기준점의 x좌표
        y (float): 기준점의 y좌표
        angle (float): 라디안 단위의 회전 각도
        r (float): 이동할 거리

    Returns:
        (float, float): 새로운 좌표 (x_new, y_new)
    """
    # angle = angle - np.pi

    x_new = position[0] + r * math.sin(angle)
    y_new = position[1] + r * math.cos(angle)
    return np.array([x_new, y_new, position[2]])

def get_offset_xy_position(object_name:str, position):
    # {"cuboid": 0, "cylinder": 1, "hexagonal_prism": 2, "needle": 3, "torus": 4, "tube": 5 }
    offset_xy = position

    if object_name == "cuboid":
        pass
    elif object_name == "cylinder":
        pass
    elif object_name == "hexagonal_prism":
        pass
    elif object_name == "needle":
        pass
    elif object_name == "torus":
        r = 0.05 / get_stage_units()
        # offset_xy = compute_new_coordinate(position, angle, r)
    elif object_name == "tube":
        pass
    else:
        pass

    return offset_xy