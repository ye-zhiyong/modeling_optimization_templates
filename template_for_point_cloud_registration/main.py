import numpy as np
import pathlib
import open3d as o3d
import matplotlib.pyplot as plt
import os
import base_gps
from pyproj import Proj, Transformer
from scipy.spatial.transform import Rotation

# Point Cloud Registration
class RegistrationPCD:
    def __init__(self, data) -> None:
        self.pcd_file_list = list(map(str, list(pathlib.Path(data['pcd_folder']).glob("*.pcd"))))
        self.pcd_file_list.sort()
        self.gps = base_gps.base_gps(data['gps_file'])
        self.save_path = os.path.join(data['save_folder'], 'stitch_pcd_result.pcd')

    # get millisecond-level timestamps of pcd and gps
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
    
    # get GPS coordinate of every point cloud frame
    def get_pcd_gps(self):
        # get pcd and gps's millisecond level timestamps
        gps_time_msec, pcd_time_msec = self.get_time_msec()
        
        # match a set of GPS coordinate points(the front, middle and rear points) for each point cloud frame
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
    
    # point cloud registration
    def registration(self):
        # get GPS coordinate of pcds
        pcd_gps = self.get_pcd_gps()
        assert len(pcd_gps) == len(self.pcd_file_list), "GPS数据与点云帧数不匹配"
        
        # get UTM coordinate transformer
        utm_zone = int((np.mean([p[0] for p in pcd_gps]) + 180)/6) + 1
        wgs84_to_utm_2d = Transformer.from_crs("EPSG:4326", f"EPSG:326{utm_zone}")

        # get reference UTM coordinate
        ref_lon, ref_lat, ref_alt = pcd_gps[1]
        easting, northing= wgs84_to_utm_2d.transform(ref_lat, ref_lon)
        ref_utm = np.array([easting, northing, ref_alt])
        
        # point clouds registration
        registered_pcd = o3d.geometry.PointCloud()
        voxel_size = 0.3
        height_values = []  
        num = 0
        #begin_index, end_index = self.get_valid_pcd_index()
        #for i in range(begin_index, end_index):
        for i in range(len(self.pcd_file_list)):
            # get pcd and downsampling
            pcd = o3d.io.read_point_cloud(self.pcd_file_list[i])
            pcd_down = pcd.voxel_down_sample(voxel_size)
            
            # get UTM coordinate of every point cloud frame
            lon, lat, alt = pcd_gps[i]
            easting, northing = wgs84_to_utm_2d.transform(lat, lon)
            curr_utm = np.array([easting, northing, alt])
            print(curr_utm)
            
            # get transform matrix
            T = np.eye(4)
            T[:3, 3] = curr_utm - ref_utm
            T_inv = np.linalg.inv(T)
            
            # coordinate transform 
            pcd_down.transform(T_inv) 
            registered_pcd += pcd_down

            num += 1
            print(f"已完成第 {num} 帧点云配准")
        
        # filter
            # elevation filter
        z_min = -1
        z_max = 20
        z_coords = np.asarray(registered_pcd.points)[:, 2]
        valid_mask = (z_coords >= z_min) & (z_coords <= z_max)
        registered_pcd = registered_pcd.select_by_index(np.where(valid_mask)[0])
            # noise filter
        registered_pcd, _ = registered_pcd.remove_statistical_outlier(nb_neighbors = 20, std_ratio = 1.0)

        # set color based on the height of point cloud
        if len(registered_pcd.points) > 0:
            points = np.asarray(registered_pcd.points)
            height_values = points[:, 2] 
            min_h, max_h = np.min(height_values), np.max(height_values)
            colors = plt.cm.jet((height_values - min_h) / (max_h - min_h))[:, :3]
            registered_pcd.colors = o3d.utility.Vector3dVector(colors)

        # reserve registered point clouds
        o3d.io.write_point_cloud(self.save_path, registered_pcd)
        
        # visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="配准结果-高度着色", width=1000, height=800)
        vis.add_geometry(registered_pcd)
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
        vis.add_geometry(mesh_frame)
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        vis.run()
        vis.destroy_window()

        return registered_pcd

if __name__ == '__main__':
    data = {
        'pcd_folder': r"C:\Users\ap\Desktop\WorkSpace\point_cloud_registration\point_cloud_registration_algorithm_v3.0\point_cloud_dataset\202501111557",
        'gps_file': r"C:\Users\ap\Desktop\WorkSpace\point_cloud_registration\point_cloud_registration_algorithm_v3.0\point_cloud_dataset\202501111557\GPS.txt",
        'save_folder': r'C:\Users\ap\Desktop\WorkSpace\point_cloud_registration\point_cloud_registration_algorithm_v3.0'
    }
    rp = RegistrationPCD(data)
    rp.registration()