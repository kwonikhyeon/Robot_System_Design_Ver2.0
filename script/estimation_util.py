from scipy.spatial import KDTree
import os
import numpy as np

from omni.isaac.sensor import Camera
from omni.isaac.core import World

DEBUG = False

def quaternion_from_axis_angle(axis, angle):
    axis = np.array(axis) / np.linalg.norm(axis)
    w = np.cos(angle / 2)
    xyz = np.sin(angle / 2) * axis
    return np.array([w, *xyz])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def random_downsample_pointcloud(pointcloud, num_points):
    if len(pointcloud) <= num_points:
        print("Point cloud has fewer points than requested. Returning original.")
        return pointcloud

    # 랜덤으로 인덱스를 선택
    selected_indices = np.random.choice(len(pointcloud), num_points, replace=False)
    downsampled_pointcloud = pointcloud[selected_indices]
    return downsampled_pointcloud

def load_pointcloud(file_path):
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

def get_correspondence_indices_3d(P, Q, distance_threshold=0.5):
    tree = KDTree(Q.T)  # KDTree 생성
    distances, indices = tree.query(P.T)  # P의 각 점에 대해 가장 가까운 Q의 점을 찾음
    correspondences = []
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if dist <= distance_threshold:  # 거리 제한 추가
            correspondences.append((i, idx))
    return correspondences

def compute_cross_covariance_3d(P, Q, correspondences, kernel=lambda diff: 1.0):
    P_points = P[:, [i for i, _ in correspondences]]
    Q_points = Q[:, [j for _, j in correspondences]]
    diffs = Q_points - P_points
    weights = kernel(diffs)  # 가중치 계산
    cov = (weights * Q_points) @ P_points.T
    return cov

def icp_rotation_only(P, Q, iterations=200, threshold=1e-10, sigma=1.0):
    norm_values = []
    R_final = np.eye(3)  # Initialize final rotation matrix as identity
    t_final = np.zeros((3, 1))  # 초기 Translation 벡터

    for i in range(iterations):
        P_transformed = R_final.dot(P) + t_final

        correspondences = get_correspondence_indices_3d(P_transformed, Q)
        
        cov = compute_cross_covariance_3d(P_transformed, Q, correspondences,kernel=lambda diff: gaussian_kernel(diff, sigma=sigma))

        U, S, V_T = np.linalg.svd(cov)
        R = U.dot(V_T)

        # 회전 적용 후 Translation 계산
        P_rotated = R.dot(P_transformed)
        t = Q[:, [j for _, j in correspondences]].mean(axis=1, keepdims=True) - P_rotated.mean(axis=1, keepdims=True)
        
        R_final = R.dot(R_final)
        t_final = R.dot(t_final) + t

        # 대응점에 대해 매칭된 점들만 오차 계산
        aligned_diff = np.array([np.linalg.norm(P_rotated[:, i] + t - Q[:, j]) 
                                 for i, j in correspondences])
        current_norm = np.mean(aligned_diff)
        norm_values.append(current_norm)

        if i > 0 and abs(norm_values[-1] - norm_values[-2]) < threshold:
            if DEBUG:
                print(f"Converged at iteration {i} with change {abs(norm_values[-1] - norm_values[-2])} < threshold {threshold}")
            break

    P_aligned = R_final.dot(P) + t_final
    return P_aligned, R_final, t_final

def gaussian_kernel(diff, sigma=1.0):
    distances_squared = np.sum(diff**2, axis=0)
    return np.exp(-distances_squared / (2 * sigma**2))

def align_centroids(P, Q):
    # 각 데이터셋의 중심 계산
    center_Q = np.zeros((3, 1))
    center_P = np.zeros((3, 1))
    
    center_Q[0] = max(Q[0]) - (np.linalg.norm(max(Q[0]) - min(Q[0]))) / 2.0
    center_Q[1] = max(Q[1]) - (np.linalg.norm(max(Q[1]) - min(Q[1]))) / 2.0
    center_Q[2] = max(Q[2]) - (np.linalg.norm(max(Q[2]) - min(Q[2]))) / 2.0
    center_P[0] = max(P[0]) - (np.linalg.norm(max(P[0]) - min(P[0]))) / 2.0
    center_P[1] = max(P[1]) - (np.linalg.norm(max(P[1]) - min(P[1]))) / 2.0
    center_P[2] = max(P[2]) - (np.linalg.norm(max(P[2]) - min(P[2]))) / 2.0
    
    # 중심 이동을 통해 데이터 정렬
    P_centered = P - center_P
    Q_centered = Q - center_Q
    
    return P_centered, Q_centered, center_P, center_Q

def get_filtered_pointcloud(camera:Camera, object_segmentation_id):
    # 현재 프레임 가져오기
    current_frame = camera.get_current_frame()

    # 세그멘테이션 및 뎁스 데이터 가져오기
    segmentation = current_frame["instance_segmentation"]["data"]  # 세그멘테이션 데이터
    depth_image = current_frame["distance_to_image_plane"]  # 뎁스 데이터
    resolution = segmentation.shape  # 이미지 해상도

    # 데이터 정렬 확인 (세그멘테이션과 뎁스 크기 일치)
    if segmentation.shape != depth_image.shape:
        raise ValueError("Segmentation and depth image resolutions do not match!")

    # 세그멘테이션 마스크 생성 (관심 있는 객체만 필터링)
    object_mask = segmentation == object_segmentation_id

    # 깊이 데이터 유효성 확인 (0 또는 NaN 값 필터링)
    valid_depth_mask = (depth_image > 0) & ~np.isnan(depth_image)

    # 최종 유효 마스크 (세그멘테이션 + 깊이 유효성)
    final_mask = object_mask & valid_depth_mask

    # 깊이 데이터와 매핑할 2D 이미지 좌표 생성
    height, width = resolution
    xx = np.linspace(0, width - 1, width)
    yy = np.linspace(0, height - 1, height)
    xmap, ymap = np.meshgrid(xx, yy)  # 2D 이미지 좌표
    image_coords = np.column_stack((xmap.ravel(), ymap.ravel()))  # (n, 2) 형식

    # 마스크 적용하여 관심 있는 객체의 좌표만 선택
    filtered_image_coords = image_coords[final_mask.ravel()]
    filtered_depth = depth_image[final_mask]

    # 관심 있는 포인트를 카메라 좌표계에서 세계 좌표계로 변환
    world_points = camera.get_world_points_from_image_coords(filtered_image_coords, filtered_depth)

    return world_points

def get_object_id(camera, target_class_name):
    current_frame = camera.get_current_frame()
    instance_segmentation_info = current_frame["instance_segmentation"]["info"]["idToSemantics"]
     
    for object_id, label_info in instance_segmentation_info.items():
        if label_info["class"] == target_class_name:
            return int(object_id)

    # 찾지 못한 경우 경고 메시지 출력
    print(f"[WARNING] Object '{target_class_name}' not found in segmentation map.")
    return None

def get_camera_coordinate_at_episode(object_center:float, stage:str):
    object_position = object_center
    if stage == "Pick":
        base = 0.25
        height = 0.45
        cam_angle = np.arctan2(height, base)
        q1_1 = quaternion_from_axis_angle([0, 0, 1], np.pi)
        q1_2 = quaternion_from_axis_angle([0, 1, 0], cam_angle)
        q1 = quaternion_multiply(q1_1, q1_2)

        q2_1 = quaternion_from_axis_angle([0, 0, 1], -np.pi / 2)
        q2_2 = quaternion_from_axis_angle([0, 1, 0], cam_angle)
        q2 = quaternion_multiply(q2_1, q2_2)

        q3_1 = quaternion_from_axis_angle([0, 0, 1], 0)
        q3_2 = quaternion_from_axis_angle([0, 1, 0], cam_angle)
        q3 = quaternion_multiply(q3_1, q3_2)

        q4_1 = quaternion_from_axis_angle([0, 0, 1], np.pi / 2)
        q4_2 = quaternion_from_axis_angle([0, 1, 0], cam_angle)
        q4 = quaternion_multiply(q4_1, q4_2)

        q5 = quaternion_from_axis_angle([0, 0, 1], np.pi)

        q6 = quaternion_from_axis_angle([0, 0, 1], -np.pi / 2)

        q7 = quaternion_from_axis_angle([0, 0, 1], 0)

        q8 = quaternion_from_axis_angle([0, 0, 1], np.pi / 2)

        camera_orientations = [q1, q2, q3, q4, q5, q6, q7, q8]

        camera_positions = [
        object_position + np.array([base, 0.0, height]),
        object_position + np.array([0.0, base, height]),
        object_position + np.array([-base, 0.0, height]),
        object_position + np.array([0.0, -base, height]),
        object_position + np.array([base, 0.0, 0.05]),
        object_position + np.array([0.0, base, 0.05]),
        object_position + np.array([-base, 0.0, 0.05]),
        object_position + np.array([0.0, -base, 0.05]),
        ]
        if DEBUG:
            print(f"cam_angle:{cam_angle}")
            print(f"pick pos: {camera_positions}")
    
        return camera_positions, camera_orientations
    elif stage == "Put":
        base = 0.25
        height = 0.25
        cam_angle = np.arctan2(height, base)
        q1_1 = quaternion_from_axis_angle([0, 0, 1], np.pi)
        q1_2 = quaternion_from_axis_angle([0, 1, 0], cam_angle)
        q1 = quaternion_multiply(q1_1, q1_2)

        q2_1 = quaternion_from_axis_angle([0, 0, 1], -np.pi / 2)
        q2_2 = quaternion_from_axis_angle([0, 1, 0], cam_angle)
        q2 = quaternion_multiply(q2_1, q2_2)

        q3_1 = quaternion_from_axis_angle([0, 0, 1], 0)
        q3_2 = quaternion_from_axis_angle([0, 1, 0], cam_angle)
        q3 = quaternion_multiply(q3_1, q3_2)

        q4_1 = quaternion_from_axis_angle([0, 0, 1], np.pi / 2)
        q4_2 = quaternion_from_axis_angle([0, 1, 0], cam_angle)
        q4 = quaternion_multiply(q4_1, q4_2)

        q5_1 = quaternion_from_axis_angle([0, 0, 1], np.pi)
        q5_2 = quaternion_from_axis_angle([0, 1, 0], -cam_angle)
        q5 = quaternion_multiply(q5_1, q5_2)

        q6_1 = quaternion_from_axis_angle([0, 0, 1], -np.pi / 2)
        q6_2 = quaternion_from_axis_angle([0, 1, 0], -cam_angle)
        q6 = quaternion_multiply(q6_1, q6_2)

        q7_1 = quaternion_from_axis_angle([0, 0, 1], 0)
        q7_2 = quaternion_from_axis_angle([0, 1, 0], -cam_angle)
        q7 = quaternion_multiply(q7_1, q7_2)

        q8_1 = quaternion_from_axis_angle([0, 0, 1], np.pi / 2)
        q8_2 = quaternion_from_axis_angle([0, 1, 0], -cam_angle)
        q8 = quaternion_multiply(q8_1, q8_2)

        camera_orientations = [q1, q2, q3, q4, q5, q6, q7, q8]

        camera_positions = [
        object_position + np.array([base, 0.0, height]),
        object_position + np.array([0.0, base, height]),
        object_position + np.array([-base, 0.0, height]),
        object_position + np.array([0.0, -base, height]),
        object_position + np.array([base, 0.0, -height]),
        object_position + np.array([0.0, base, -height]),
        object_position + np.array([-base, 0.0, -height]),
        object_position + np.array([0.0, -base, -height])
        ]
        if DEBUG:
            print(f"cam_angle:{cam_angle}")
            print(f"put pos: {camera_positions}")

        return camera_positions, camera_orientations
    else:
        print(f"stage is unvaild: {stage}")
        return [], []

def rotation_matrix_to_euler_angles(matrix):
    assert matrix.shape == (3, 3), "Input must be a 3x3 matrix."
    
    # Calculate Yaw (Z-axis rotation)
    yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    # Calculate Pitch (Y-axis rotation)
    pitch = np.arcsin(-matrix[2, 0])
    # Calculate Roll (X-axis rotation)
    roll = np.arctan2(matrix[2, 1], matrix[2, 2])

    if DEBUG:
        print(f"roll, pitch, yaw: {np.rad2deg(roll)}, {np.rad2deg(pitch)}, {np.rad2deg(yaw)}")
    
    return roll, pitch, yaw

def get_actual_coordinate(my_world:World, my_camera:Camera, object_name:str, object_center:float, stage:str):
    save_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"images/{object_name}")
    os.makedirs(save_root, exist_ok=True)
    if stage == "Pick":
        object_center_2d = np.array([object_center[0][0], object_center[0][1], 0.0])
    elif stage == "Put":
        object_center_2d = np.array([object_center[0][0], object_center[0][1], object_center[0][2]])
    else:
        raise ValueError(f"Unknown stage: {stage}")
    
    camera_positions, camera_orientations = get_camera_coordinate_at_episode(object_center_2d, stage)
    
    merged_pcl = []

    print("Pose Estimating...")
    for i in range(8):
        my_camera.set_world_pose(camera_positions[i], camera_orientations[i], "world")

        for _ in range(10):  # 몇 프레임 더 진행하여 월드 업데이트 보장
            my_world.step(render=True)     

        # object에 대한 세그멘테이션 ID 추출
        object_id = get_object_id(my_camera, object_name)

        # 관심 객체에 대한 필터링된 포인트 클라우드 가져오기
        filtered_pointcloud = get_filtered_pointcloud(my_camera, object_id)

        if DEBUG:
            print(f"Episode: {i}")
            if object_id is not None:
                print(f"Segmentation ID for '{object_name}': {object_id}")
            else:
                print(f"'{object_name}'의 세그멘테이션 ID를 찾을 수 없습니다.")
            print(f"Filtered point cloud size: {filtered_pointcloud.shape}")

        merged_pcl.append(filtered_pointcloud)
    
    # 병합
    merged_pcl = np.vstack(merged_pcl)
    # 중복 제거
    merged_pcl_rounded = np.round(merged_pcl, decimals=4)  # 정밀도 조정
    merged_pcl_unique = np.unique(merged_pcl_rounded, axis=0)

    pcl_file = os.path.join(save_root, f"{object_name}_pointcloud.npy")
    object_pcl = load_pointcloud(pcl_file)

    P_prealign, Q_prealign, P_center, Q_center= align_centroids(merged_pcl_unique.T, object_pcl.T)

    # 다운샘플링 수행
    num_points_to_keep = 5000 # 다운샘플링 후 남길 포인트 수
    downsampled_object = random_downsample_pointcloud(P_prealign.T, num_points_to_keep)
    downsampled_pcl = random_downsample_pointcloud(Q_prealign.T, num_points_to_keep*2)

    P_aligned, R_final, T_final= icp_rotation_only(downsampled_object.T, downsampled_pcl.T)

    T_output = P_center + T_final

    roll, pitch, yaw = rotation_matrix_to_euler_angles(R_final.T)
    R_output = np.array([roll, pitch, yaw])

    # 가장 유사한 축 찾기
    # closest_axis, similarity = find_closest_axis_to_world_z(R_final.T)
    # print(f"가장 유사한 로컬 축 인덱스: {closest_axis}")
    # print(f"회전각: {rotation_axis[closest_axis]:.3f}")

    return T_output.reshape(-1), R_output

