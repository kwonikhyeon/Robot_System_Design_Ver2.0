from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.tasks import BaseTask
from omni.isaac.core.utils.prims import is_prim_path_valid, create_prim, delete_prim
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import add_reference_to_stage, get_stage_units
from omni.isaac.core.utils.prims import create_prim, delete_prim, define_prim, get_prim_path
from omni.isaac.franka import Franka
from omni.isaac.cortex.cortex_utils import get_assets_root_path
from omni.isaac.core.utils.semantics import add_update_semantics
from omni.isaac.sensor import Camera
import omni.isaac.core.utils.numpy.rotations as rot_utils

from typing import Optional
from scipy.spatial.transform import Rotation
import numpy as np
import os

class PickInsertion(BaseTask):
    def __init__(self, name: str = "pick_insertion") -> None:
        BaseTask.__init__(self, name=name, offset=None)
        self._fr3 = None
        self._holes = []
        self._objects = []

        if os.name =='nt':
            filepath = os.path.abspath(__file__)
            split = filepath.split('\\')
            root_path = split[0]
            for name in split[1:9]:
                root_path = root_path + '\\' + name
            self._assets_root_path = root_path
            print(root_path)
        else:
            filepath = os.path.abspath(__file__)
            split = filepath.split('/')
            root_path = ''
            for name in split[1:9]:
                root_path = root_path + '/' + name
            self._assets_root_path = root_path   
            print(root_path)
            
        self._fr3_asset_name = "fr3.usd"
        self._object_asset_name = ['cuboid', 'cylinder', 'hexagonal_prism', 'needle', 'torus', 'tube']
        self._hole_asset_name = ["hole_" + name for name in self._object_asset_name]
                     
        return

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        scene.add_default_ground_plane()        

        # Add single robot
        self._fr3 = scene.add(
            Franka(
                prim_path="/World/robots/fr3_0",
                name="fr3_0",
                position=[0.0, 0.0, 0.0],
                gripper_open_position=np.array([0.1, 0.1]) / get_stage_units()
            )
        )

        # Add objects
        if os.name =='nt':
            add_reference_to_stage(usd_path=self._assets_root_path + "\\objects", prim_path="/World/objects")
        else:
            add_reference_to_stage(usd_path=self._assets_root_path + "/objects", prim_path="/World/objects")
        for i in range(6):
            translation = np.zeros(3)
            translation[0] = np.random.uniform(low=-0.2, high=0.2)     # 20cm variation
            translation[1] = np.random.uniform(low=-0.3, high=0.3)     # 30cm variation
            translation += np.array([0.4, 0.3, 0.2])   # drop objects from 20cm height
            orientation = np.random.uniform(low=0.0, high=1.0, size=4)
            orientation = orientation / np.linalg.norm(orientation)
            if os.name =='nt':
                object_asset_path = self._assets_root_path + "\\Robot_System_Design_Ver2.0\\objects\\" + self._object_asset_name[i] + ".usd"
            else:
                object_asset_path = self._assets_root_path + "/Robot_System_Design_Ver2.0/objects/" + self._object_asset_name[i] + ".usd"

            add_reference_to_stage(usd_path=object_asset_path, prim_path="/World/objects")
            self._objects.append(
                scene.add(
                    RigidPrim(
                        prim_path="/World/objects/" + self._object_asset_name[i],
                        name=self._object_asset_name[i],
                        translation=translation,
                        orientation=orientation,
                        mass=0.001
                    )
                )
            )
        
        # Add holes
        index = 0        
        for i in range(2):
            for j in range(3):
                if os.name =='nt':
                    hole_asset_path = self._assets_root_path + "\\Robot_System_Design_Ver2.0\\objects\\" + self._hole_asset_name[index] + ".usd"
                else:
                    hole_asset_path = self._assets_root_path + "/Robot_System_Design_Ver2.0/objects/" + self._hole_asset_name[index] + ".usd"
                   
                add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/objects")
                self._holes.append(
                    scene.add(
                        XFormPrim(
                            prim_path="/World/objects/" + self._hole_asset_name[index],
                            name=self._hole_asset_name[index],
                            translation=[0.5-0.13*i, -0.3-0.13+0.13*j, 0.0],
                            orientation=[1.0, 0.0, 0.0, 0.0]
                        )
                    )
                )
                index += 1
        
        # Update semantics
        for i in range(6):            
            add_update_semantics(prim=self._objects[i].prim, semantic_label=self._object_asset_name[i])
            add_update_semantics(prim=self._holes[i].prim, semantic_label=self._hole_asset_name[i])
            
        # Camera setting
        self._camera = scene.add(
            Camera(
                prim_path="/World/camera",
                name="cam_0",
                frequency=30,
                resolution=(1920, 1080),
                orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True)
            )
        )    
        ori = Rotation.from_euler('z', angles=-90, degrees=True).as_quat() # (x,y,z,w) in scipy
        ori = ori[[3, 0, 1, 2]] # (w,x,y,z) in Isaac scipy
        self._camera.set_world_pose(position=[0.5, 0.0, 2.0], orientation=ori, camera_axes='usd')        
        self._camera.initialize()
        self._camera.set_focal_length(2)
        self._camera.add_distance_to_camera_to_frame()
        self._camera.add_instance_segmentation_to_frame()
        self._camera.add_pointcloud_to_frame() 
        
        return
    
    def get_observations(self) -> dict:
        """Returns current observations from the task."""
        joint_state = self._fr3.get_joints_state()
        return {
            self._fr3.name: {
                "joint_positions": np.array(joint_state.positions),
                "gripper_joint_positions": np.array(self._fr3.gripper.get_joint_positions()),
            },
        }

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        BaseTask.pre_step(self, time_step_index=time_step_index, simulation_time=simulation_time)
        return

    def post_reset(self) -> None:
        return

    def cleanup(self) -> None:
        return

    def get_params(self) -> dict:
        params_representation = dict()
        params_representation["robot_name"] = {"value": [self._fr3.name], "modifiable": False}
        params_representation["camera_name"] = {"value": self._camera.name, "modifiable": False}
        return params_representation
