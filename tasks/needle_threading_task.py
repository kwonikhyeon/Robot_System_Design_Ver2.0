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

from typing import Optional
# from panda.pandaKinematics import pandaKinematics
from scipy.spatial.transform import Rotation
import numpy as np
import os

class NeedleThreading(BaseTask):
    def __init__(self, name: str = "needle_threading") -> None:
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


        self._assets_root_path = root_path        
        self._fr3_asset_name = "fr3.usd"
        self._object_asset_name = "needle_threading"
        self._hole_asset_name = ["hole_threading1", "hole_threading2"]

        return

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        scene.add_default_ground_plane()        

        # Add robots
        self._fr3 =[
            scene.add(Franka(prim_path="/World/robots/fr3_0", name="fr3_0", position=[0.0, 0.3, 0.0])),
            scene.add(Franka(prim_path="/World/robots/fr3_1", name="fr3_1", position=[0.0, -0.3, 0.0]))
        ]

        # Add objects
        translation = np.array([0.5, 0.0, 0.1])   # drop objects from 10cm height
        orientation = np.random.uniform(low=0.0, high=1.0, size=4)
        orientation = orientation / np.linalg.norm(orientation)
        if os.name == "nt":
            object_asset_path = self._assets_root_path + "\\Robot_System_Design_Ver2.0\\objects\\" + self._object_asset_name + ".usd"
        else:
            object_asset_path = self._assets_root_path + "/Robot_System_Design_Ver2.0/objects/" + self._object_asset_name + ".usd"
        add_reference_to_stage(usd_path=object_asset_path, prim_path="/World/objects")
        self._objects = scene.add(
            RigidPrim(
                prim_path="/World/objects/" + self._object_asset_name,
                name=self._object_asset_name,
                translation=translation,
                orientation=orientation
                )
            )
        
        # Add holes
        for i in range(2):
            if os.name == "nt":
                hole_asset_path = self._assets_root_path + "\\Robot_System_Design_Ver2.0\\objects\\" + self._hole_asset_name[i] + ".usd"
            else:
                hole_asset_path = self._assets_root_path + "/Robot_System_Design_Ver2.0/objects/" + self._hole_asset_name[i] + ".usd"
            add_reference_to_stage(usd_path=hole_asset_path, prim_path="/World/objects")
            self._holes.append(
                scene.add(
                    XFormPrim(
                        prim_path="/World/objects/" + self._hole_asset_name[i],
                        name=self._hole_asset_name[i],
                        translation=[0.5, -0.5+1.0*i, 0.0],
                        orientation=[1.0, 0.0, 0.0, 0.0]
                        )
                    )
                )
        
        # Update semantics
        add_update_semantics(prim=self._objects.prim, semantic_label=self._object_asset_name)
        add_update_semantics(prim=self._holes[0].prim, semantic_label=self._hole_asset_name[0])
        add_update_semantics(prim=self._holes[1].prim, semantic_label=self._hole_asset_name[1])
        
        # Camera setting
        self._camera = scene.add(
            Camera(
                prim_path="/World/camera",
                name="cam_0",
                frequency=30,
                resolution=(1920, 1080),
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
        """Returns current observations from the task needed for the behavioral layer at each time step.

           Observations:            
            - robot:
                - joint_positions
                - end_effector_pose

        Returns:
            dict: [description]
        """
        joint_state_0 = self._fr3[0].get_joints_state()
        joint_state_1 = self._fr3[1].get_joints_state()
        # T_0 = pandaKinematics.fk(joints=joint_state_0.positions[:7])[0][-1]
        # T_1 = pandaKinematics.fk(joints=joint_state_1.positions[:7])[0][-1]
                
        # The end_effector frame is attached at hand, not the distal tip
        # position_0, orientation_0 = self._fr3[0].end_effector.get_local_pose()  # orientation in quat (w,x,y,z)
        # position_1, orientation_1 = self._fr3[1].end_effector.get_local_pose()
                
        return {
                 self._fr3[0].name: {
                    "joint_positions": np.array(joint_state_0.positions),
                    # "pose": np.array(T_0)
                },
                self._fr3[1].name: {
                    "joint_positions": np.array(joint_state_1.positions),
                    # "pose": np.array(T_1)
                },
            }


    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """Executed before the physics step.

        Args:
            time_step_index (int): Current time step index
            simulation_time (float): Current simulation time.
        """
        BaseTask.pre_step(self, time_step_index=time_step_index, simulation_time=simulation_time)
        return

    def post_reset(self) -> None:
        """Executed after reseting the scene"""
        return

    def cleanup(self) -> None:
        """Removed the added screws when resetting."""
        return

    def get_params(self) -> dict:
        """Task parameters are
            - robot_name

        Returns:
            dict: defined parameters of the task.
        """
        params_representation = dict()
        params_representation["robot_name"] = {"value": [self._fr3[0].name, self._fr3[1].name], "modifiable": False}
        params_representation["camera_name"] = {"value": self._camera.name, "modifiable": False}
        return params_representation

      