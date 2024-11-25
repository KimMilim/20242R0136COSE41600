# 03_eleminate_road_point
# 시각화에 필요한 라이브러리 불러오기
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# pcd 파일 불러오기, 필요에 맞게 경로 수정
file_path = "test_data/1727320101-665925967.pcd"
# PCD 파일 읽기
original_pcd = o3d.io.read_point_cloud(file_path)

# Voxel Downsampling 수행
voxel_size = 0.2  # 필요에 따라 voxel 크기를 조정
downsample_pcd = original_pcd.voxel_down_sample(voxel_size=voxel_size)


# 밀도 기반으로 SOR 및 ROR 매개변수 동적 설정
avg_density = len(original_pcd.points) / original_pcd.get_axis_aligned_bounding_box().volume()

# SOR 매개변수 조정
sor_nb_neighbors = max(10, int(avg_density * 5))
sor_std_ratio = 1.0 if avg_density > 0.1 else 0.8

# ROR 매개변수 조정
ror_nb_points = max(5, int(avg_density * 3))
ror_radius = 1.0 if avg_density > 0.1 else 0.8

# Statistical Outlier Removal (SOR) 적용
cl, ind = downsample_pcd.remove_statistical_outlier(nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio)
sor_pcd = downsample_pcd.select_by_index(ind)

# Radius Outlier Removal (ROR) 적용
cl, ind = sor_pcd.remove_radius_outlier(nb_points=ror_nb_points, radius=ror_radius)
ror_pcd = sor_pcd.select_by_index(ind)

# RANSAC을 사용하여 평면 추정
plane_model, inliers = ror_pcd.segment_plane(distance_threshold=0.25,
                                             ransac_n=3,
                                             num_iterations=1000)

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

# 도로에 속하는 포인트 (inliers)
road_pcd = ror_pcd.select_by_index(inliers)

# 도로에 속하지 않는 포인트 (outliers)
non_road_pcd = ror_pcd.select_by_index(inliers, invert=True)

# 도로 영역을 초록색으로 표시
road_pcd.paint_uniform_color([1, 0, 0])  # 빨간색으로 표시
# 도로가 아닌 포인트를 초록색으로 표시
non_road_pcd.paint_uniform_color([0, 1, 0])  # 녹색으로 표시

# 포인트 클라우드 시각화 함수
def visualize_point_clouds(pcd_list, window_name="Point Cloud Visualization", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    for pcd in pcd_list:
        vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

# 두 영역을 동시에 시각화 (포인트 크기를 원하는 크기로 조절 가능)
visualize_point_clouds([road_pcd, non_road_pcd], 
                       window_name="Road (Red) and Non-Road (Green) Points", point_size=2.0)

