import numpy as np
import pathlib
import open3d as o3d
import matplotlib.pyplot as plt
import os
from base_gps import base_gps
from pyproj import Transformer



class RegistrationPCD:
    def __init__(self, data):
        self.pcd_file_list = list(map(str, list(pathlib.Path(data['pcd_folder']).glob("*.pcd"))))
        self.pcd_file_list.sort()
        self.gps = base_gps(data['gps_file'])
        self.save_path = os.path.join(data['save_folder'], 'stitch_pcd_result.pcd')
        # 检查保存目录是否存在，不存在则创建
        os.makedirs(data['save_folder'], exist_ok=True)
        # 检查路径合法性
        if not os.access(data['save_folder'], os.W_OK):
            raise PermissionError(f"无权限写入目录: {data['save_folder']}")
        self.z_min = data.get('z_min', -1)
        self.z_max = data.get('z_max', 20)
        self.voxel_size = data.get('voxel_size', 0.3)

    def get_time_msec(self):
        gps_time_msec = np.array(
            [
                int(i[:2]) * 3600000 + int(i[3:5]) * 60000 + int(i[6:8]) * 1000 + int(i[9:])
                for i in self.gps.data["uav_time"].tolist()
            ]
        )
        pcd_time_msec = np.array(
            [
                int(i[-13:-11]) * 3600000 + int(i[-11:-9]) * 60000 + int(i[-9:-7]) * 1000 + int(i[-7:-4])
                for i in self.pcd_file_list
            ]
        )
        return gps_time_msec, pcd_time_msec
    
    def get_pcd_gps(self):
        gps_time_msec, pcd_time_msec = self.get_time_msec()
        
        pcd_gps = []
        for i in pcd_time_msec:
            gps_index = np.argmin(np.abs(gps_time_msec - i))
            gps_index = 1 if gps_index < 1 else gps_index
            gps_index = len(self.gps.data) - 2 if gps_index > len(self.gps.data) - 2 else gps_index
            pcd_gps.append(
                [self.gps.data["longitude"][gps_index - 1],
                 self.gps.data["latitude"][gps_index - 1],
                 self.gps.data['altitude'][gps_index - 1]]
            )
        return pcd_gps
    
    def registration(self):
        pcd_gps = self.get_pcd_gps()
        assert len(pcd_gps) == len(self.pcd_file_list), "GPS数据与点云帧数不匹配"
        
        utm_zone = int((np.mean([p[0] for p in pcd_gps]) + 180)/6) + 1
        wgs84_to_utm_2d = Transformer.from_crs("EPSG:4326", f"EPSG:326{utm_zone}")

        ref_lon, ref_lat, ref_alt = pcd_gps[1]
        easting, northing= wgs84_to_utm_2d.transform(ref_lat, ref_lon)
        ref_utm = np.array([easting, northing, ref_alt])
        
        registered_pcd = o3d.geometry.PointCloud()
        num = 0
        
        for i in range(len(self.pcd_file_list)):
            pcd = o3d.io.read_point_cloud(self.pcd_file_list[i])
            pcd_down = pcd.voxel_down_sample(self.voxel_size)
            
            lon, lat, alt = pcd_gps[i]
            easting, northing = wgs84_to_utm_2d.transform(lat, lon)
            curr_utm = np.array([easting, northing, alt])
            
            T = np.eye(4)
            T[:3, 3] = curr_utm - ref_utm
            T_inv = np.linalg.inv(T)
            
            pcd_down.transform(T_inv) 
            registered_pcd += pcd_down
            num += 1
        
        # 高程滤波
        z_coords = np.asarray(registered_pcd.points)[:, 2]
        valid_mask = (z_coords >= self.z_min) & (z_coords <= self.z_max)
        registered_pcd = registered_pcd.select_by_index(np.where(valid_mask)[0])
        
        # 去噪
        registered_pcd, _ = registered_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.0)

        # 高度着色
        if len(registered_pcd.points) > 0:
            points = np.asarray(registered_pcd.points)
            height_values = points[:, 2] 
            min_h, max_h = np.min(height_values), np.max(height_values)
            colors = plt.cm.jet((height_values - min_h) / (max_h - min_h))[:, :3]
            registered_pcd.colors = o3d.utility.Vector3dVector(colors)

        # 保存结果
        o3d.io.write_point_cloud(self.save_path, registered_pcd, write_ascii=True)
        
        return registered_pcd

    def visualize(self, pcd):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="配准结果-高度着色", width=1000, height=800)
        vis.add_geometry(pcd)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        vis.add_geometry(mesh_frame)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        vis.run()
        vis.destroy_window()