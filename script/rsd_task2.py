import os
import sys
from ultralytics import YOLO

# 현재 스크립트 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))

# 상위 폴더 경로 가져오기
parent_dir = os.path.dirname(current_dir)

# 상위 폴더를 sys.path에 추가
sys.path.append(parent_dir)

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import carb
from omni.isaac.core import World
from omni.isaac.core.utils.types import ArticulationAction
# from tasks.pick_insertion_task import PickInsertion
from tasks.needle_threading_task import NeedleThreading
from omni.isaac.franka import KinematicsSolver
from scipy.spatial.transform import Rotation
import omni.isaac.core.utils.numpy.rotations as rot_utils
from matplotlib import pyplot as plt
import numpy as np

from grasp_controller import GraspController # 익현(수정 예정)
import detection_util as detection # 규진
import estimation_util as estimation # 지호

import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.utils.rotations import euler_angles_to_quat


## global variable
shapes_object = {3: "needle_threading", 4: "needle_mark"}
shapes_hole = {1: "hole_threading1", 2: "hole_threading2"}
object_number = {"needle_threading": 3, "needle_mark": 4}

# Detection result path
object_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "object_images")
hole_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hole_images")
os.makedirs(object_path, exist_ok=True)
os.makedirs(hole_path, exist_ok=True)

# World setting
my_world = World(stage_units_in_meters=1.0)
my_task = NeedleThreading()
my_world.add_task(my_task)
my_world.reset()

# Get instance
franka_name = {0: my_task.get_params()["robot_name"]["value"][0], 1: my_task.get_params()["robot_name"]["value"][1]}
my_franka0 = my_world.scene.get_object(franka_name[0])
my_franka1 = my_world.scene.get_object(franka_name[1])
# print(franka_name)

# fr3_0
my_controller0 = GraspController(
    name="grasp_controller0", gripper=my_franka0.gripper, robot_articulation=my_franka0, world=my_world
)
articulation_controller0 = my_franka0.get_articulation_controller()

# fr3_1
my_controller1 = GraspController(
    name="grasp_controller1", gripper=my_franka1.gripper, robot_articulation=my_franka1, world=my_world
)
articulation_controller1 = my_franka1.get_articulation_controller()

camera_name = my_task.get_params()["camera_name"]["value"]
my_camera = my_world.scene.get_object(camera_name)

# Initialize the World and Camera
my_world.reset()
detection.camera_init(my_camera)
my_franka0.gripper.set_default_state(my_franka0.gripper.joint_opened_positions)
my_franka1.gripper.set_default_state(my_franka1.gripper.joint_opened_positions)
my_world.scene.add_default_ground_plane()

my_world.reset()

#===FUNCTION========================================================================================
def activate_phase1():
    init_joint = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    default_orientation = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
    tolerance = 0.1

    # franka0 robot joint init
    articulation_controller0.apply_action(
        ArticulationAction(joint_positions=init_joint)
    )

    if np.all(np.abs(my_franka0.get_joint_positions() - init_joint) <= tolerance):
        
        # Needle Object 판단
        my_camera.set_world_pose([0.5, 0, 0.55], default_orientation)
    
        for _ in range(5):  # 몇 프레임 더 진행하여 월드 업데이트 보장
            my_world.step(render=True)
        
        current_frame = my_camera.get_current_frame()
        distance_image = current_frame["distance_to_image_plane"]
        detection.save_depth_image(distance_image, os.path.join(hole_path, "test"), 2)

        result_object = detection.getInfoNeedle(my_camera, distance_image)

        print("\n=== OBJECT ===")
        for i in result_object:
            print(f"{shapes_object[i]}: ", result_object[i])

            # 좌표 확인용            
            prims_utils.create_prim(
                prim_path=f"/World/Object_Centroid_{shapes_object[i]}",
                prim_type="Xform",
                position=result_object[i][0]
            )


        result_hole = {}
        # return : [0: hole_threading 1, 1: hole_threading 2]

        # hole_threading1 판단
        my_camera.set_world_pose([0.5, -0.5, 0.6], default_orientation)
        
        for _ in range(5):  # 몇 프레임 더 진행하여 월드 업데이트 보장
            my_world.step(render=True)
        
        current_frame = my_camera.get_current_frame()
        distance_image = current_frame["distance_to_image_plane"]
        detection.save_depth_image(distance_image, os.path.join(hole_path, "test"), 3)

        result_hole[0] = detection.getInfoHole_threading(my_camera, distance_image)
        
        # hole_threading2 판단
        my_camera.set_world_pose([0.5, 0.5, 0.6], default_orientation)
                       
        for _ in range(5):  # 몇 프레임 더 진행하여 월드 업데이트 보장
            my_world.step(render=True)

        current_frame = my_camera.get_current_frame()
        distance_image = current_frame["distance_to_image_plane"]
        detection.save_depth_image(distance_image, os.path.join(hole_path, "test"), 3)

        result_hole[1] = detection.getInfoHole_threading(my_camera, distance_image)
        
        print("\n=== HOLE ===")
        for i in result_hole:
            print(f"{shapes_hole[i+1]}: ", result_hole[i])

            # 좌표 확인용            
            prims_utils.create_prim(
                prim_path=f"/World/Hole_Centroid_{shapes_hole[i+1]}",
                prim_type="Xform",
                position=result_hole[i][0]
            )         
        print("\n")

        return 2, result_object, result_hole
    return 1, [], []


def activate_phase2(target_object, target_center, stage):
    position, rotation= estimation.get_actual_coordinate(my_world, my_camera, target_object, target_center, stage)

    print("\n=== Pose Estimation Result ===")
    print(f"[object]: {target_object}")
    print(f"[position]\n{position}")
    print(f"[rotation]\n{rotation}")

    # 좌표 확인용            
    prims_utils.create_prim(
        prim_path=f"/World/estimated_{target_object}",
        prim_type="Xform",
        position=np.array(position),
        orientation=euler_angles_to_quat(rotation)
    )  
    return 3, position, rotation

def activate_phase3(object_name:str, position, rotation):
    my_controller0.target_object = object_name
    my_controller0.world = my_world
    my_controller0.camera = my_camera

    # my_franka.gripper.set_joint_closed_positions(np.array([0.1, 0.1]))
    
    actions = my_controller0.forward(        
            picking_position=position,
            placing_position=np.array([-0.3, -0.3, 0.0515 / 2.0]),
            current_joint_positions=my_franka0.get_joint_positions(),
            object_rotation=rotation
            # end_effector_offset=np.array([0, 0.005, 0]),
            # end_effector_orientation = rotation,
        )
    
    articulation_controller0.apply_action(actions)
#     # print(my_controller.get_current_event())


#===================================================================================================

# Simulation loop
global target_object

phase = 1 
stage = "None"

my_camera.add_motion_vectors_to_frame()
reset_needed = False

# cvtColor Error 방지
is_semantic_initialized = False
while not is_semantic_initialized:
    my_world.step(render=True)
    if my_camera.get_current_frame()["instance_segmentation"] is not None:
        is_semantic_initialized = True
my_world.reset()

for _ in range(100):  # 몇 프레임 더 진행하여 월드 업데이트 보장
    my_world.step(render=True)

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_needed = False
        # Get observations
        # observations = my_world.get_observations()
        
        # q = observations[franka_name]['joint_positions']
        # gq = observations[franka_name]['gripper_joint_positions']
                    
        if phase == 1: # object detection
            phase, object_center, hole_center = activate_phase1()   
            target_number = object_number["needle_threading"]
            target_object = shapes_object[target_number]
            target_center = object_center[target_number] if object_center else None
            stage = "Pick"

        # if phase == 2: # pose estimation
        #     phase, target_position, target_orientation = activate_phase2(target_object, target_center, stage)

        # if phase == 3: # first pass (franka 0)
        #     activate_phase3(target_object, target_position, target_orientation) # TODO: fix controller

        # if phase == 4: # first handover (franka 0, 1)
            # print()
        
        # if phase == 5: # second pass (franka 1)
            # print()
        
        # if phase == 6: # seond handover (franka 0, 1)
            # print()

simulation_app.close()