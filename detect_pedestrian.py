import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import argparse

# ROR SOR 
def remove_outliers_sor_ror(pcd, method, **kwargs):
    if method == "SOR":
        cl, ind = pcd.remove_statistical_outlier(**kwargs)
    elif method == "ROR":
        cl, ind = pcd.remove_radius_outlier(**kwargs)
    return pcd.select_by_index(ind)

def visualize_point_clouds(pcd_list, window_name="Point Cloud Visualization", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()
    
def capture_and_visualize_pcd(pcd, bounding_boxes, file_name, point_size=0.5, zoom_factor=0.8):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud")
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    
    vis.get_render_option().point_size = point_size
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom_factor)  
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_name) 
    vis.destroy_window()

    
    
# 경로 입력 받아 파싱
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--scenario', type=str, required=True)
args = parser.parse_args()

# PCD 파일이 저장된 디렉토리 경로
data_dir = args.data_dir
senario_name = args.scenario
pcd_dir = os.path.join(data_dir, senario_name, "pcd/")

# 디렉토리 확인
if not os.path.exists(pcd_dir):
    raise FileNotFoundError(f"PCD directory not found: {pcd_dir}")

# PCD 파일 리스트 생성
pcd_files = [os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir) if f.endswith('.pcd')]
results = []

for file_path in pcd_files:
    print(f"Processing: {file_path}")
    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)
    
    # Voxel Downsampling
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=0.1)
    
    # 밀도 기반으로 SOR 및 ROR 매개변수 동적 설정
    avg_density = len(original_pcd.points) / original_pcd.get_axis_aligned_bounding_box().volume()
    # print(f"avg_density={avg_density:.2f}")
    
    # SOR ROR 매개변수 조정
    if avg_density < 1.5:  # 밀도가 낮을 경우
        sor_nb_neighbors = max(30, int(avg_density * 10))  
        sor_std_ratio = 2.0  
        ror_nb_points = max(10, int(avg_density * 5))
        ror_radius = 2.0 
    else:  # 밀도가 높을 경우
        sor_nb_neighbors = max(20, int(avg_density * 5))
        sor_std_ratio = 1.0
        ror_nb_points = max(5, int(avg_density * 3))
        ror_radius = 1.0
    
    # print(f"sor_nb_neighbors = {sor_nb_neighbors:.2f},sor_std_ratio = {sor_std_ratio:.2f}")
    # print(f"ror_nb_points = {ror_nb_points:.2f},ror_radius = {ror_radius:.2f}")
    
    #visualize_point_clouds([downsample_pcd])
    
    # SOR ROR
    sor_pcd = remove_outliers_sor_ror(downsample_pcd, method="SOR", nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio)
    ror_pcd = remove_outliers_sor_ror(sor_pcd, method="ROR", nb_points= ror_nb_points, radius=ror_radius)

    # RANSAC으로 평면 추출
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000) # n=10
    final_point = ror_pcd.select_by_index(inliers, invert=True)
    
     
    # 밀도를 기반으로 최소 포인트 설정
    avg_density = len(final_point.points) / final_point.get_axis_aligned_bounding_box().volume()
    min_points = min(10, int(avg_density * 10))
    # print(f"final_avg_den={avg_density}, min_points={min_points}")
    
    # DBSCAN 클러스터링
    labels = np.array(final_point.cluster_dbscan(eps=0.37, min_points=min_points, print_progress=True)) 
    max_label = labels.max()
    # print(f"Point cloud has {max_label + 1} clusters")

    colors = plt.get_cmap("tab20")(labels/(max_label if max_label>0 else 1))
    colors[labels<0] = 0
    final_point.colors = o3d.utility.Vector3dVector(colors[:,:3])
 
    # 필터링 조건
    min_points_in_cluster = 10
    max_points_in_cluster = 120 
    min_z_value = -1.5
    max_z_value = 1.0
    min_height = 0.1
    max_height = 1.0
    max_distance = 200.0
    min_width = 0.03  # 최소 가로 길이  
    max_width = 1.0  # 최대 가로 길이  
    min_aspect_ratio = 2.0  # 최소 z/max(x,y) 비율
    max_aspect_ratio = 5.0  # 최대 z/max(x,y) 비율 
    
    # 조건에 따라 바운딩 박스 생성
    bboxes = []
    for i in range(max_label + 1):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            
            # 높이 필터링
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()
            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    
                     # x, y 값의 범위 계산 (가로 길이)
                    x_min, x_max = points[:, 0].min(), points[:, 0].max()
                    y_min, y_max = points[:, 1].min(), points[:, 1].max()
                    width_x = x_max - x_min
                    width_y = y_max - y_min
                    
                    if height_diff <= max(width_x,width_y):
                        continue

                    # 가로 길이와 비율 조건 추가
                    if min_width <= width_x <= max_width and min_width <= width_y <= max_width:
                        aspect_ratio = height_diff / max(width_x, width_y)  # 비율 계산 
                        if min_aspect_ratio <= aspect_ratio and aspect_ratio <=  max_aspect_ratio:  # 비율 조건 확인
                        
                            # 거리 조건 필터링
                            distances = np.linalg.norm(points, axis=1)
                            if distances.max() <= max_distance:
                                bbox = cluster_pcd.get_axis_aligned_bounding_box()
                                bbox.color = (1, 0, 0)
                                bboxes.append(bbox)
                                # Z축 범위 출력
                                # print(f"Cluster {i}: z_min = {z_min:.2f}, z_max = {z_max:.2f}, z_diff = {z_max-z_min:.2f}")

    
    # 시각화
    # visualize_with_bounding_boxes(final_point, bboxes, point_size=1.0)
    
    # 처리 결과 저장
    results.append((file_path, final_point, bboxes))


# 캡처된 이미지를 저장할 디렉토리
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

# 각 파일의 결과를 캡처
frame_files = []
for idx, (file_path, final_point, bboxes) in enumerate(results):
    frame_file = os.path.join(output_dir, f"frame_{idx:03d}.png")
    print(f"Capturing: {frame_file}")
    capture_and_visualize_pcd(final_point, bboxes, frame_file, point_size=1.0)
    frame_files.append(frame_file)


video_file = senario_name+"_output_video.mp4"
frame_rate = 5  # 초당 프레임 수 
frame_size = None

if frame_files:
    temp_image = cv2.imread(frame_files[0])
    frame_size = (temp_image.shape[1], temp_image.shape[0])

if frame_size:
    # 동영상 생성기 초기화
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, frame_size)

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {video_file}")