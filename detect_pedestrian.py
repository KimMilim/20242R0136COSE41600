import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

# PCD 시나리오 디렉토리가 저장된 상위 경로
base_dir = "data"
scenario_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# 필터링 조건 설정
voxel_size = 0.1
min_points_in_cluster = 5
max_points_in_cluster = 40
min_z_value = -1.5
max_z_value = 2.5
min_height = 0.5
max_height = 2.0
max_distance = 30.0

# 시각화 및 동영상 생성 함수
def capture_and_visualize_pcd(pcd, bounding_boxes, file_name, point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud")
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_name)  # 캡처된 화면 저장
    vis.destroy_window()

def process_scenario(scenario_dir):
    print(f"Processing scenario: {scenario_dir}")
    pcd_files = [os.path.join(scenario_dir, f) for f in os.listdir(scenario_dir) if f.endswith('.pcd')]
    
    # 캡처된 이미지를 저장할 디렉토리
    output_dir = os.path.join(scenario_dir, "output_frames")
    os.makedirs(output_dir, exist_ok=True)

    # 결과 저장
    results = []
    frame_files = []

    for file_path in pcd_files:
        print(f"Processing PCD: {file_path}")
        # PCD 파일 읽기
        original_pcd = o3d.io.read_point_cloud(file_path)
        
        # Voxel Downsampling
        downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)
        
        # Statistical Outlier Removal
        cl, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)
        sor_pcd = downsample_pcd.select_by_index(ind)
        
        # RANSAC으로 평면 추출
        plane_model, inliers = sor_pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=2000)
        final_point = sor_pcd.select_by_index(inliers, invert=True)
        
        # DBSCAN 클러스터링
        labels = np.array(final_point.cluster_dbscan(eps=0.3, min_points=10, print_progress=True))
        max_label = labels.max()
        print(f"Point cloud has {max_label + 1} clusters")
        
        # 각 클러스터 색상 지정
        colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
        colors[labels < 0] = 0  # 노이즈는 검정색
        final_point.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        # 조건에 따라 바운딩 박스 생성
        bboxes = []
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
                cluster_pcd = final_point.select_by_index(cluster_indices)
                points = np.asarray(cluster_pcd.points)
                z_values = points[:, 2]
                z_min = z_values.min()
                z_max = z_values.max()
                if min_z_value <= z_min and z_max <= max_z_value:
                    height_diff = z_max - z_min
                    if min_height <= height_diff <= max_height:
                        distances = np.linalg.norm(points, axis=1)
                        if distances.max() <= max_distance:
                            bbox = cluster_pcd.get_axis_aligned_bounding_box()
                            bbox.color = (1, 0, 0)
                            bboxes.append(bbox)
        
        # 처리 결과 저장
        results.append((file_path, final_point, bboxes))
        
        # 시각화 및 캡처
        frame_file = os.path.join(output_dir, f"{os.path.basename(file_path)}.png")
        capture_and_visualize_pcd(final_point, bboxes, frame_file, point_size=2.0)
        frame_files.append(frame_file)

    # OpenCV로 동영상 생성
    video_file = os.path.join(scenario_dir, "output_video.mp4")
    frame_rate = 2  # 초당 프레임 수
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

# 각 시나리오를 처리
for scenario_dir in scenario_dirs:
    process_scenario(scenario_dir)
