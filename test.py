import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# ROR SOR 
def remove_outliers_sor_ror(pcd, method, **kwargs):
    
    if method == "SOR":
        cl, ind = pcd.remove_statistical_outlier(**kwargs)
    elif method == "ROR":
        cl, ind = pcd.remove_radius_outlier(**kwargs)
    return pcd.select_by_index(ind)

# 포인트 클라우드 및 바운딩 박스를 시각화하는 함수
def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()
    
# 포인트 클라우드 시각화 함수
def visualize_point_clouds(pcd_list, window_name="Point Cloud Visualization", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()
    
# PCD 파일이 저장된 디렉토리 경로
# pcd_dir = "data/02_straight_duck_walk/pcd/"
pcd_dir = "data/"

senario_name = "04_zigzag_walk"
pcd_dir = pcd_dir + senario_name+"/pcd/"

pcd_files = [os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir) if f.endswith('.pcd')]

# 결과 저장
results = []

for file_path in pcd_files:
    print(f"Processing: {file_path}")
    # PCD 파일 읽기
    original_pcd = o3d.io.read_point_cloud(file_path)
    
    # 밀도 기반으로 SOR 및 ROR 매개변수 동적 설정
    # 단위 부피 당 point 2.09
    avg_density = len(original_pcd.points) / original_pcd.get_axis_aligned_bounding_box().volume()

    print(f"avg_density={avg_density:.2f}")
    
    # SOR 매개변수 조정
    if avg_density < 1.5:  # 밀도가 낮을 경우
        sor_nb_neighbors = max(30, int(avg_density * 10))  # 더 많은 이웃 점
        sor_std_ratio = 2.0  # 노이즈 제거 강도 완화
    else:  # 밀도가 높을 경우
        sor_nb_neighbors = max(20, int(avg_density * 5))
        sor_std_ratio = 1.0
    
    print(f"sor_nb_neighbors = {sor_nb_neighbors:.2f},sor_std_ratio = {sor_std_ratio:.2f}")

    # ROR 매개변수 조정
    if avg_density < 1.5:  # 밀도가 낮을 경우
        ror_nb_points = max(10, int(avg_density * 5))  # 더 많은 이웃 점 필요
        ror_radius = 2.0  # 반경 증가
    else:  # 밀도가 높을 경우
        ror_nb_points = max(5, int(avg_density * 3))
        ror_radius = 1.0
    
    print(f"ror_nb_points = {ror_nb_points:.2f},ror_radius = {ror_radius:.2f}")
    

    # Voxel Downsampling
    downsample_pcd = original_pcd.voxel_down_sample(voxel_size=0.1)
    
    #visualize_point_clouds([downsample_pcd])
    
    # SOR ROR
    sor_pcd = remove_outliers_sor_ror(downsample_pcd, method="SOR", nb_neighbors=20, std_ratio=1.3) # 10,2.0
    ror_pcd = remove_outliers_sor_ror(sor_pcd, method="ROR", nb_points= 5, radius=1.5) # 6, 1.5


    # RANSAC으로 평면 추출
    plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000) # n=10
    final_point = ror_pcd.select_by_index(inliers, invert=True)
    
     
    # 밀도를 기반으로 최소 포인트 설정
    avg_density = len(final_point.points) / final_point.get_axis_aligned_bounding_box().volume()
    min_points = min(10, int(avg_density * 10))
    
    
    print(f"final_avg_den={avg_density}, min_points={min_points}")
    
    # DBSCAN 클러스터링
    labels = np.array(final_point.cluster_dbscan(eps=0.37, min_points=min_points, print_progress=True))  # eps 0.45

    max_label = labels.max()
    print(f"Point cloud has {max_label + 1} clusters")
    
    # # 각 클러스터 색상 지정
    # colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    # colors[labels < 0] = 0  # 노이즈는 검정색
    # final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])
    
    # # 노이즈 포인트는 검정색, 클러스터 포인트는 파란색으로 지정
    # colors = np.zeros((len(labels), 3))  # 기본 검정색 (노이즈)
    # colors[labels >= 0] = [0, 0, 1]  # 파란색으로 지정
    # final_point.colors = o3d.utility.Vector3dVector(colors)

    colors = plt.get_cmap("tab20")(labels/(max_label if max_label>0 else 1))
    colors[labels<0] = 0
    final_point.colors = o3d.utility.Vector3dVector(colors[:,:3])
    
    
    
    # 필터링 조건
    min_points_in_cluster = 10
    max_points_in_cluster = 120 # 80
    min_z_value = -1.5
    max_z_value = 1.0
    min_height = 0.1
    max_height = 1.0
    max_distance = 200.0
    min_width = 0.03  # 최소 가로 길이  # 0.1
    max_width = 1.0  # 최대 가로 길이  # 1.0
    min_aspect_ratio = 2.0  # 최소 z/max(x,y) 비율
    max_aspect_ratio = 5.0  # 최대 z/max(x,y) 비율 # 4
    
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

    
    # 시각화 (포인트 크기를 원하는 크기로 조절 가능)
    # visualize_with_bounding_boxes(final_point, bboxes, point_size=1.0)
    
    # 처리 결과 저장
    results.append((file_path, final_point, bboxes))

# 시각화 함수
def capture_and_visualize_pcd(pcd, bounding_boxes, file_name, point_size=0.5, zoom_factor=0.8):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud")
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    
    # 렌더링 옵션 설정
    vis.get_render_option().point_size = point_size
    
    # 카메라 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom_factor)  # 줌을 설정하여 확대
    
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_name)  # 캡처된 화면 저장
    vis.destroy_window()

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

# OpenCV로 동영상 생성
#video_file = f"{scenario_name}_video.mp4"
video_file = senario_name+"_output_video.mp4"
frame_rate = 5  # 초당 프레임 수 -> 5 로 마지막에 바꾸기 
frame_size = None

# 첫 번째 이미지에서 프레임 크기 가져오기
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