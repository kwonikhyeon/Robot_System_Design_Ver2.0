# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import typing

import os
import numpy as np
import cv2
import math

import grasp_util as gu

from scipy.spatial.transform import Rotation as R
from omni.isaac.core import World
from omni.isaac.sensor import Camera
from omni.isaac.core.controllers.base_controller import BaseController
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.manipulators.grippers.gripper import Gripper

from omni.isaac.franka.controllers.rmpflow_controller import RMPFlowController
from omni.isaac.manipulators.grippers.parallel_gripper import ParallelGripper
import omni.isaac.core.utils.numpy.rotations as rot_utils

## TODO: attatch camera on gripper
##       seq 1. move to top of object
##       seq 2. get top view of object
##       seq 3. find new grasp position(using point cloud)
##       seq 3-1. calculate x,y and rotation of gripper (write code by each object)
##       seq 3-2. calculate z(half of height between ground and highset point of object)
##       seq 4. move to grasp position(and move down)
##       seq 5. grasp object
##       seq 6. move up
##              end

class GraspController(BaseController):
    """
    A simple pick and place state machine for tutorials

    Each phase runs for 1 second, which is the internal time of the state machine

    Dt of each phase/ event step is defined

    - Phase 0: Move end_effector above the cube center at the 'end_effector_initial_height'.
    - Phase 1: Lower end_effector down to encircle the target cube
    - Phase 2: Wait for Robot's inertia to settle.
    - Phase 3: close grip.
    - Phase 4: Move end_effector up again, keeping the grip tight (lifting the block).
    - Phase 5: Smoothly move the end_effector toward the goal xy, keeping the height constant.
    - Phase 6: Move end_effector vertically toward goal height at the 'end_effector_initial_height'.
    - Phase 7: loosen the grip.
    - Phase 8: Move end_effector vertically up again at the 'end_effector_initial_height'
    - Phase 9: Move end_effector towards the old xy position.

    Args:
        name (str): Name id of the controller
        cspace_controller (BaseController): a cartesian space controller that returns an ArticulationAction type
        gripper (Gripper): a gripper controller for open/ close actions.
        end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from (more info in phases above). If not defined, set to 0.3 meters. Defaults to None.
        events_dt (typing.Optional[typing.List[float]], optional): Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

    Raises:
        Exception: events dt need to be list or numpy array
        Exception: events dt need have length of 10
    """

    def __init__(
        self,
        name: str,
        # cspace_controller: BaseController,
        gripper: Gripper,
        robot_articulation: Articulation,
        world: typing.Optional[World] = None,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,        
    ) -> None:
        BaseController.__init__(self, name=name)
        self.target_object = ""
        
        self.world = world
        self.camera = None
        self.arm = robot_articulation

        self._captured = False
        self._angle_for_grasp = np.zeros((3, 1))
        self._event = 0
        self._t = 0
        self._h1 = end_effector_initial_height
        if self._h1 is None:
            self._h1 = 0.3 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            ## fix event dt to dynamic
            self._events_dt = [0.01, 0.01, 0.3, 0.1, 0.005, 0.05, 0.0025, 1, 0.008, 0.08]
        else:
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("events dt need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        self._cspace_controller = RMPFlowController(
                name=name + "_cspace_controller", robot_articulation=robot_articulation
        )
        self._gripper = gripper
        self._pause = False
        return

    def is_paused(self) -> bool:
        """

        Returns:
            bool: True if the state machine is paused. Otherwise False.
        """
        return self._pause

    def get_current_event(self) -> int:
        """

        Returns:
            int: Current event/ phase of the state machine
        """
        return self._event

    def forward(
        self,        
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        object_rotation:np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Runs the controller one step for the current phase.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray): The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): Offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): End effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])

        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        gripper_position, gripper_orientation = self._gripper.get_world_pose()

        if not np.all(self._angle_for_grasp == np.zeros((3, 1))):
            end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi, self._angle_for_grasp[2]]))

        closest_axis, similarities= gu.find_closest_axis_to_world_z(object_rotation)                
        is_standing = gu.is_object_standing(self.target_object, closest_axis)
        
        # Phase 0: Move above the picking position
        if self._event == 0:                  
            if np.all(self._gripper.get_joint_positions()) <= 0.001:                
                print(f"closest_axis: {closest_axis}, is_standing: {is_standing}")
                self._angle_for_grasp = object_rotation
                for idx, angle in enumerate(object_rotation):
                    if angle >= 0:
                        self._angle_for_grasp[idx] = angle - np.pi / 2.0
                    else:
                        self._angle_for_grasp[idx] = angle + np.pi / 2.0
                if self.target_object == "needle" or self.target_object == "needle_threading" or self.target_object == "torus":
                    self._angle_for_grasp[2] = object_rotation[2] + np.pi / 2.0

                target_joint_positions = self._gripper.forward(action="open")
            
            else:
                offset_position = gu.get_offset_xy_position(self.target_object, picking_position)
                
                self._current_target_x = offset_position[0]
                self._current_target_y = offset_position[1]
                self._h0 = offset_position[2]
                position_target = np.array([
                    self._current_target_x + end_effector_offset[0],
                    self._current_target_y + end_effector_offset[1],
                    self._h1 + end_effector_offset[2],
                ])
                target_joint_positions = self._cspace_controller.forward(
                    target_end_effector_position=position_target,
                    target_end_effector_orientation=end_effector_orientation,
                )

            diff = np.abs(gripper_position - np.array([self._current_target_x, self._current_target_y, self._h1]))
            if np.all(diff < 0.05):
                for _ in range(10):  # 몇 프레임 더 진행하여 월드 업데이트 보장
                    target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
                    self.world.step(render=True) 
                self._event += 1

        # Phase 1: Lower down to encircle the object
        elif self._event == 1:  # Phase 1: Lower down to encircle the object and open the gripper                                                                                  
            a = self._mix_sin(max(0, self._t))
            target_height = self._combine_convex(self._h1, self._h0, a)
            target_height = gripper_position[2]-0.075
            position_target = np.array([
                self._current_target_x + end_effector_offset[0],
                self._current_target_y + end_effector_offset[1],
                target_height + end_effector_offset[2],
            ])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )

            if ((gripper_position[2]-0.035) - gu.get_offset_z_position(self.target_object, picking_position[2], is_standing)) < 0.005:
                self._event += 1            

        # Phase 2: Wait for robot's inertia to settle
        elif self._event == 2:            
            self._captured = False
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
            
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        # Phase 3: Close the gripper to grasp the object
        elif self._event == 3:            
            target_joint_positions = self._gripper.forward(action="close")

            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        # Phase 4: Lift the object
        elif self._event == 4:            
            lift_height = 0.6         
            a = self._mix_sin(max(0, self._t))
            target_height = gripper_position[2] + 0.125
            position_target = np.array([
                self._current_target_x + end_effector_offset[0],
                self._current_target_y + end_effector_offset[1],
                target_height + end_effector_offset[2],
            ])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )        

            if (abs(lift_height - (gripper_position[2]))) < 0.05:
                self._event += 1            

        # Phase 5: Move horizontally toward the placing position
        elif self._event == 5: 
            lift_height = 0.6                                                                     
            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi/2, 0]))
            position_target = np.array([
                interpolated_xy[0] + end_effector_offset[0],
                interpolated_xy[1] + end_effector_offset[1],
                lift_height + end_effector_offset[2],
            ])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )

            # 남은 진행 정도(1 - self._t)에 비례해서 dt를 줄이기
            remaining = 1.0 - self._t
            if remaining < 0.0001:
                remaining = 1.0
            increment = self._events_dt[self._event] * remaining

            self._t += increment

            if self._t >= 1.0:                
                for _ in range(10):  # 몇 프레임 더 진행하여 월드 업데이트 보장
                    target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
                    self.world.step(render=True) 
                self._t = 0
                self._event += 1
        
        elif self._event == 6:
            return target_joint_positions

        # Invalid phase
        else:
            raise ValueError(f"Unknown event phase: {self._event}")

        return target_joint_positions

    def insert(
        self,        
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        object_rotation:np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Runs the controller one step for the current phase.

        Args:
            picking_position (np.ndarray): The object's position to be picked in local frame.
            placing_position (np.ndarray): The object's position to be placed in local frame.
            current_joint_positions (np.ndarray): Current joint positions of the robot.
            end_effector_offset (typing.Optional[np.ndarray], optional): Offset of the end effector target. Defaults to None.
            end_effector_orientation (typing.Optional[np.ndarray], optional): End effector orientation while picking and placing. Defaults to None.

        Returns:
            ArticulationAction: Action to be executed by the ArticulationController
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])

        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if end_effector_orientation is None:
            end_effector_orientation = euler_angles_to_quat(np.array([0, np.pi, 0]))
        
        gripper_position, gripper_orientation = self._gripper.get_world_pose()

        # Phase 6: Lower down to the placing height
        if self._event == 6:
            a = self._mix_sin(self._t)
            target_height = self._combine_convex(self._h1, placing_position[2], a)
            position_target = np.array([
                placing_position[0] + end_effector_offset[0],
                placing_position[1] + end_effector_offset[1],
                target_height + end_effector_offset[2],
            ])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )

            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0


        # Phase 7: Open the gripper to release the object
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action="open")

            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0


        # Phase 8: Move vertically up again
        elif self._event == 8:
            a = self._mix_sin(self._t)
            target_height = self._combine_convex(placing_position[2], self._h1, a)
            position_target = np.array([
                placing_position[0] + end_effector_offset[0],
                placing_position[1] + end_effector_offset[1],
                target_height + end_effector_offset[2],
            ])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )

            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0


        # Phase 9: Return to the initial XY position
        elif self._event == 9:
            position_target = np.array([
                self._current_target_x + end_effector_offset[0],
                self._current_target_y + end_effector_offset[1],
                self._h1 + end_effector_offset[2],
            ])
            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target,
                target_end_effector_orientation=end_effector_orientation,
            )
            self._t += self._events_dt[self._event]
            if self._t >= 1.0:
                self._event += 1
                self._t = 0

        # Invalid phase
        else:
            raise ValueError(f"Unknown event phase: {self._event}")

        return target_joint_positions

    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        xy_target = (1 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        if self._event < 5:
            return 0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 1
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h1
        else:
            raise ValueError()
        return h

    def _mix_sin(self, t):
        return 0.5 * (1 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1 - alpha) * a + alpha * b
 
    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Resets the state machine to start from the first phase/ event

        Args:
            end_effector_initial_height (typing.Optional[float], optional): end effector initial picking height to start from. If not defined, set to 0.3 meters. Defaults to None.
            events_dt (typing.Optional[typing.List[float]], optional):  Dt of each phase/ event step. 10 phases dt has to be defined. Defaults to None.

        Raises:
            Exception: events dt need to be list or numpy array
            Exception: events dt need have length of 10
        """
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, np.ndarray) and not isinstance(self._events_dt, list):
                raise Exception("event velocities need to be list or numpy array")
            elif isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        return

    def is_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase. Otherwise False.
        """
        if self._event >= len(self._events_dt):
            return True
        else:
            return False

    def is_forword_done(self) -> bool:
        """
        Returns:
            bool: True if the state machine reached the last phase of forword(6). Otherwise False.
        """
        if self._event == 6:
            return True
        else:
            return False


    def pause(self) -> None:
        """Pauses the state machine's time and phase."""
        self._pause = True
        return

    def resume(self) -> None:
        """Resumes the state machine's time and phase."""
        self._pause = False
        return