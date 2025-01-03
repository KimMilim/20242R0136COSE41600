import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import argparse
import time

# ROR SOR 
def remove_outliers_sor_ror(pcd, method, **kwargs):
    if method == "SOR":
        cl, ind = pcd.remove_statistical_outlier(**kwargs)
    elif method == "ROR":
        cl, ind = pcd.remove_radius_outlier(**kwargs)
    return pcd.select_by_index(ind)

def visualize_with_bounding_boxes(pcd, bounding_boxes, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name,width=1200, height=800)
    vis.clear_geometries()
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

def capture_pcd_and_bboxes(pcd, bboxes, file_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True,window_name="Point Cloud",width=1200, height=800)
    
    lookat = np.array([-10,0, 20])  # 카메라가 바라볼 중심
    front = np.array([0, -1, 1])  # 카메라의 앞 방향 (z 축 음수 방향)
    up = np.array([0, 1, 0])      # 카메라의 위 방향 (y 축 방향)
    zoom = 0.2
    
    vis.add_geometry(pcd)
    for bbox in bboxes:
        vis.add_geometry(bbox)

    vis.get_render_option().point_size = 0.1
    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)
    
    # vis.run()
    # 시각화 및 캡쳐
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(file_name) 
    print(f"capture {file_name}")
    vis.destroy_window()
    
def object_check(pcd, bbox, file_name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True,window_name="Point Cloud",width=1200, height=800)
    
    lookat = np.array([-10,0, 20])  # 카메라가 바라볼 중심
    front = np.array([0, -1, 1])  # 카메라의 앞 방향 (z 축 음수 방향)
    up = np.array([0, 1, 0])      # 카메라의 위 방향 (y 축 방향)
    zoom = 0.2
    
    vis.add_geometry(pcd)
    vis.add_geometry(bbox)

    vis.get_render_option().point_size = 0.1
    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)
    
    vis.run()
    # 시각화 및 캡쳐
    vis.poll_events()
    vis.update_renderer()
    #vis.capture_screen_image(file_name) 
    print(f"object check {file_name}")
    vis.destroy_window()
    
def create_video_from_images(frame_files, output_file, frame_rate=5):
    if not frame_files:
        print("No frames to create video.")
        return

    # 첫 번째 이미지를 읽어와서 영상 크기 결정
    temp_image = cv2.imread(frame_files[0])
    frame_size = (temp_image.shape[1], temp_image.shape[0])

    # 동영상 생성기 초기화
    video_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, frame_size)

    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_file}")
    
    
def get_centroid(pcd):
    points = np.asarray(pcd.points)
    return np.mean(points, axis=0)  
 
def get_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def calculate_distance_between_bboxes_vertices(bbox1, bbox2):
    # 각 바운딩 박스의 꼭짓점 좌표를 얻기
    corners1 = np.asarray(bbox1.get_box_points())  # 8개의 꼭짓점
    corners2 = np.asarray(bbox2.get_box_points())  # 8개의 꼭짓점
    
    # 각 꼭짓점 간의 거리 계산 (모든 조합을 비교)
    min_distance = float('inf')  # 초기값으로 무한대 설정
    
    for corner1 in corners1:
        for corner2 in corners2:
            distance = np.linalg.norm(corner1 - corner2)  # 유클리드 거리 계산
            min_distance = min(min_distance, distance)  # 최소 거리 업데이트
    
    return min_distance

# 경로 입력 받아 파싱
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--scenario', type=str, required=True)
args = parser.parse_args()

# PCD 파일이 저장된 디렉토리 경로
data_dir = args.data_dir
senario_name = args.scenario
# data_dir="data/"
# senario_name = "01_straight_walk"
# pcd_dir = os.path.join(data_dir, senario_name, "pcd/")
pcd_dir = os.path.join(data_dir, senario_name, "pcdback/")


# 디렉토리 확인
if not os.path.exists(pcd_dir):
    raise FileNotFoundError(f"PCD directory not found: {pcd_dir}")

# PCD 파일 리스트 생성
pcd_files = [os.path.join(pcd_dir, f) for f in os.listdir(pcd_dir) if f.endswith('.pcd')]
frame_files= []
all_pcd_centroids = []
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
    
    # DBSCAN 클러스터링                          eps=0.3이 제일 잘 됐음 시나리오 1,3
    labels = np.array(final_point.cluster_dbscan(eps=0.2, min_points=min_points, print_progress=True)) 
    max_label = labels.max()
    # print(f"Point cloud has {max_label + 1} clusters")

    colors = plt.get_cmap("tab20")(labels/(max_label if max_label>0 else 1))
    colors[labels<0] = 0
    final_point.colors = o3d.utility.Vector3dVector(colors[:,:3])
 
    # 필터링 조건
    min_points_in_cluster = 10
    min_y_value = -0.5 # y좌표 (height)
    max_y_value = 100
    min_height = 0.1
    max_height = 1.0 
    max_x_diff = 15.0
    max_z_diff=20.0
    max_y_diff = 1.0
    min_distance = 1.0
    
    
    # 조건에 따라 바운딩 박스 생성
    bboxes = []
    cur_pcd_centroids = []
    for i in range(max_label + 1): #300~600개 
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices):
            cluster_pcd = final_point.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points) # x: points[:, 0], y: points[:, 1], z(왼손좌표Z): points[:, 2]
            
            # 높이 필터링(y)
            y_values = points[:, 1] 
            y_min = y_values.min()
            y_max = y_values.max()
            y_diff=abs(y_max-y_min)
            
            if min_y_value <= y_min and y_max <= max_y_value:
                    
                x_min, x_max = points[:, 0].min(), points[:, 0].max()
                x_diff = abs(x_max-x_min)*10
                z_min, z_max = points[:, 2].min(), points[:, 2].max()
                z_diff = abs(z_max-z_min)    
                     
                if max_x_diff < x_diff or max_z_diff < z_diff:# or max_height < height_diff: 
                    continue  
                if max_y_diff < y_diff:
                    continue  
                if z_diff < 0.255 or 1.6 < z_diff:
                    continue
                
                z_x_ratio = z_diff/x_diff
                z_y_ratio = z_diff/y_diff
                if z_x_ratio < 0.05:# or z_y_ratio < 2.0:
                    continue
                # bbox = cluster_pcd.get_axis_aligned_bounding_box()
                # bbox.color = (1, 0, 0)
                # bboxes.append(bbox)
                # bbox_array=np.asarray(bbox)
                # print(f"bbox_array={bbox_array}")
                
                centroid = get_centroid(cluster_pcd) #i-th cluster의 중심
                cur_pcd_centroids.append([cluster_pcd,centroid])
                # print(f"centroid={centroid}")
                
                # print(f"x_min, x_max={x_min},{x_max}")
                # print(f"x_diff={x_diff}")
                # print(f"z_min, z_max={z_min},{z_max}")
                # print(f"z_diff={z_diff}")
                # print(f"y_min, y_max={y_min},{y_max}")
                # print(f"y_diff={y_diff}")
                
                # object_check(final_point, bbox, file_path)
                
                
                            
    # 000X의 pcd의 조건필터링 후 50~70개 중심이 cur_centroids에 저장                                         
    # print(f"cur_centroids cnt ={len(cur_centroids)}")            
    
    max_distance=-1.0
    min_distance=100.0
    if 30 <= len(all_pcd_centroids):
        
        for pcd, centroid in cur_pcd_centroids: # 현재 pcd의 필터한 거에 클러스터 하나씩 확인
            ped = True
            
            compare_pcd_centorids = all_pcd_centroids[-30]
            
            for j in range(len(compare_pcd_centorids)):
                pre_pcd, pre_centroid = compare_pcd_centorids[j]
                distance = get_distance(pre_centroid,centroid)
                # print(f"distance={distance}")
                # cur_bbox = pcd.get_axis_aligned_bounding_box()
                # pre_bbox = pre_pcd.get_axis_aligned_bounding_box()  
                # distance = calculate_distance_between_bboxes_vertices(cur_bbox, pre_bbox)
                
                # 시나리오 1,3은 1.2도 가능
                if distance < 1.2: # 보행자 아님 정지물체임 0.4 -> 0.7 -> 0.9 -> 1.0
                    ped=False
                    break
            if distance > 50: #here 10 ~50 # 50
                continue            
            if ped: #보행자이면
                bbox = pcd.get_axis_aligned_bounding_box()
                bbox.color = (1, 0, 0)
                bboxes.append(bbox)   
                #print(f"distance={distance}")   
                if min_distance > distance:
                    min_distance=distance
                if max_distance < distance:
                    max_distance=distance    
                          
    
    all_pcd_centroids.append(cur_pcd_centroids)    
             
                    
                # bbox = cluster_pcd.get_axis_aligned_bounding_box()
                # bbox.color = (1, 0, 0)
                # bboxes.append(bbox)
                # print(f"height_diff={height_diff}, x_diff={x_diff}, z_diff={z_diff}")
                # print(f"ratio(키/몸통min)={height_diff/min(x_diff,z_diff):.2f}")
                # print(f"y_min={y_min:.2f}, y_max={y_max:.2f}")
                        # object_check(final_point, bbox, file_path)
                    
                    
                    
                
    
    print(f"bboxes cnt = {len(bboxes)}")    
    print(f"min_dis={min_distance},max_dis={max_distance}")
    file_name, _ = os.path.splitext(file_path)
    prefix, suffix = file_name.split("pcd_")
    
    # 경로 확인 및 디렉토리 생성
    capture_dir = prefix+"captures/"
    if not os.path.exists(capture_dir):
        os.makedirs(capture_dir)
    capture_pcd_and_bboxes(final_point, bboxes, capture_dir+"cap_"+suffix+".png")
    frame_files.append(capture_dir+"cap_"+suffix+".png")

     

video_file = senario_name+"_output_video.mp4"
create_video_from_images(frame_files, video_file, frame_rate=5)
