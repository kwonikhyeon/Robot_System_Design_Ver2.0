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
from tasks.pick_insertion_task import PickInsertion
# from tasks.needle_threading_task import NeedleThreading
from omni.isaac.franka import KinematicsSolver
from scipy.spatial.transform import Rotation
import omni.isaac.core.utils.numpy.rotations as rot_utils
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from matplotlib import pyplot as plt
import numpy as np

from grasp_controller import GraspController # 익현(수정 예정)
import detection_util as detection # 규진
import estimation_util as estimation # 지호

import omni.isaac.core.utils.prims as prims_utils

## global variable
shapes_object = { 0: "cuboid", 1: "cylinder", 2: "hexagonal_prism", 3: "needle", 4: "torus", 5: "tube" }
shapes_hole = { 0: "hole_cuboid", 1: "hole_cylinder", 2: "hole_hexagonal", 3: "hole_needle", 4: "hole_torus", 5: "hole_tube" }

object_number = {"cuboid": 0, "cylinder": 1, "hexagonal_prism": 2, "needle": 3, "torus": 4, "tube": 5 }

# Load YOLO model
model = YOLO(f"{current_dir}/best.pt")

# Detection result path
object_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "object_images")
hole_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hole_images")
os.makedirs(object_path, exist_ok=True)
os.makedirs(hole_path, exist_ok=True)

# World setting
my_world = World(stage_units_in_meters=1.0)
my_task = PickInsertion()
my_world.add_task(my_task)
my_world.reset()

# Get instance
franka_name = my_task.get_params()["robot_name"]["value"][0]
my_franka = my_world.scene.get_object(franka_name)

my_controller = GraspController(
    name="grasp_controller", gripper=my_franka.gripper, robot_articulation=my_franka, world=my_world
)
articulation_controller = my_franka.get_articulation_controller()


camera_name = my_task.get_params()["camera_name"]["value"]
my_camera = my_world.scene.get_object(camera_name)

# Initialize the World and Camera
my_world.reset()
detection.camera_init(my_camera)
my_franka.gripper.set_default_state(my_franka.gripper.joint_opened_positions)
my_world.scene.add_default_ground_plane()

my_world.reset()

#===FUNCTION========================================================================================
def activate_phase1():
    init_joint = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    default_orientation = rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
    tolerance = 0.1
    # robot joint init
    articulation_controller.apply_action(
                    ArticulationAction(joint_positions=init_joint)
                )

    if np.all(np.abs(my_franka.get_joint_positions() - init_joint) <= tolerance):
        # Object 판단        
        my_camera.set_world_pose([0.4, 0.3, 1.0], default_orientation)
        
        for _ in range(5):  # 몇 프레임 더 진행하여 월드 업데이트 보장
            my_world.step(render=True)

        rgb_image = my_camera.get_rgba()
        current_frame = my_camera.get_current_frame()
        distance_image = current_frame["distance_to_image_plane"]
        detection.save_rgb_image(rgb_image, os.path.join(object_path, "test"))

        result_object = detection.getInfoObejct(my_camera, distance_image)
        # return : [hole 1 : cylinder, hole 2 : ...]

        print("\n=== OBJECT ===")
        for i in result_object:
            print(f"{shapes_object[i]}: ", result_object[i])

                        # 좌표 확인용            
            prims_utils.create_prim(
                prim_path=f"/World/Object_Centroid_{shapes_object[i]}",
                prim_type="Xform",
                position=result_object[i][0]
            )    
        
        # Hole 판단
        my_camera.set_world_pose([0.45, -0.35, 0.8], default_orientation)
        
        for _ in range(5):  # 몇 프레임 더 진행하여 월드 업데이트 보장
            my_world.step(render=True)

        current_frame = my_camera.get_current_frame()
        distance_image = current_frame["distance_to_image_plane"]
        detection.save_depth_image(distance_image, os.path.join(hole_path, "test"), 1)

        result_hole = detection.getInfoHole(my_camera, distance_image)
        # return : [hole 1 : cylinder, hole 2 : ...]

        print("\n=== HOLE ===")
        for i in result_hole:
            print(f"{shapes_hole[i]}: ", result_hole[i])

            # 좌표 확인용            
            prims_utils.create_prim(
                prim_path=f"/World/Hole_Centroid_{shapes_hole[i]}",
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
        prim_path=f"/World/estimated_{target_object}_PICK",
        prim_type="Xform",
        position=np.array(position),
        orientation=euler_angles_to_quat(rotation)
    )  
    return 3, position, rotation

def activate_phase3(object_name:str, position, rotation, hole_center_position):    
    my_controller.target_object = object_name
    my_controller.world = my_world
    my_controller.camera = my_camera

    if my_controller.is_forword_done():
        hand_position, hand_orientation = my_franka.end_effector.get_world_pose()        
        return 4, [hand_position], [hand_orientation]
    
    actions = my_controller.forward(        
            picking_position=position,
            placing_position=hole_center_position[0],
            current_joint_positions=my_franka.get_joint_positions(),
            object_rotation=rotation,                      
        )
    
    
    
    articulation_controller.apply_action(actions)
    return 3, [], []

def activate_phase4(target_object, target_center, stage):
    position, rotation= estimation.get_actual_coordinate(my_world, my_camera, target_object, target_center, stage)

    print("\n=== Pose Estimation Result ===")
    print(f"[object]: {target_object}")
    print(f"[position]\n{position}")
    print(f"[rotation]\n{rotation}")

    # 좌표 확인용            
    prims_utils.create_prim(
        prim_path=f"/World/estimated_{target_object}_PUT",
        prim_type="Xform",
        position=np.array(position),
        orientation=euler_angles_to_quat(rotation)
    )  
    return 5, position, rotation

def activate_phase5(object_name:str, position, rotation, hole_center_position):
    my_controller.target_object = object_name
    my_controller.world = my_world
    my_controller.camera = my_camera

    if my_controller.is_done():
        return 6

    actions = my_controller.insert(        
            picking_position=position,
            placing_position=hole_center_position[0],
            current_joint_positions=my_franka.get_joint_positions(),
            object_rotation=rotation,                      
        )
    
    articulation_controller.apply_action(actions)
    return 5


#===================================================================================================

# Simulation loop
global target_object
phase = 1 
stage = ""

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
            target_number = object_number["hexagonal_prism"] #['cuboid', 'cylinder', 'hexagonal_prism', 'needle', 'torus', 'tube']
            target_object = shapes_object[target_number]
            target_center = object_center[target_number] if object_center else None
            target_hole_center = hole_center[target_number] if hole_center else None
            stage = "Pick"

        if phase == 2: # pose estimation
            phase, target_position, target_orientation = activate_phase2(target_object, target_center, stage)

        if phase == 3: # grasp object 
            phase, hand_position, hand_orientation = activate_phase3(target_object, target_position, target_orientation, target_hole_center)             
            stage = "Put"
        if phase == 4:
            phase, grasped_object_position, grasped_object_orientation = activate_phase4(target_object, hand_position, stage)
        if phase == 5:
            pass

simulation_app.close()