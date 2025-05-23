import os
import torch
import torch.multiprocessing as mp
import torch.multiprocessing
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import open3d as o3d
import time
import rerun as rr
sys.path.append(os.path.dirname(__file__))
from argparse import ArgumentParser
from arguments import SLAMParameters
from utils.graphics_utils import focal2fov
from scene.shared_objs import SharedCam, SharedGaussians, SharedPoints, SharedTargetPoints
from gaussian_renderer import render, network_gui
from mp_Tracker import Tracker
from mp_Mapper import Mapper
from dataloader import *

torch.multiprocessing.set_sharing_strategy('file_system')

class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug
        
class GS_ICP_SLAM(SLAMParameters):
    def __init__(self, args):
        super().__init__()
        self.dataset_path = args.dataset_path
        self.config = args.config
        self.output_path = args.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = args.verbose
        self.keyframe_th = float(args.keyframe_th)
        self.knn_max_distance = float(args.knn_maxd)
        self.overlapped_th = float(args.overlapped_th)
        self.max_correspondence_distance = float(args.max_correspondence_distance)
        self.trackable_opacity_th = float(args.trackable_opacity_th)
        self.overlapped_th2 = float(args.overlapped_th2)
        self.downsample_rate = int(args.downsample_rate)
        self.test = args.test
        self.save_results = args.save_results
        self.rerun_viewer = args.rerun_viewer
        
        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.spawn(connect=False)
        
        self.dataset = self.get_dataset(self.config, self.dataset_path)
        
        self.W = int(self.dataset.image_size[0])
        self.H = int(self.dataset.image_size[1])
        self.fx = float(self.dataset.camera[0])
        self.fy = float(self.dataset.camera[1])
        self.cx = float(self.dataset.camera[2])
        self.cy = float(self.dataset.camera[3])
        self.depth_scale = self.dataset.depth_scale
        self.depth_trunc = self.dataset.depth_trunc
        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(self.downsample_rate)
        
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
        
        # Make test cam
        # To get memory sizes of shared_cam
        test_rgb_img, test_depth_img = self.get_test_image()
        test_points, _, _, _ = self.downsample_and_make_pointcloud(test_depth_img, test_rgb_img)

        # Get size of final poses
        num_final_poses = len(self.dataset)
        
        # Shared objects
        self.shared_cam = SharedCam(FoVx=focal2fov(self.fx, self.W), FoVy=focal2fov(self.fy, self.H),
                                    image=test_rgb_img, depth_image=test_depth_img,
                                    cx=self.cx, cy=self.cy, fx=self.fx, fy=self.fy)
        self.shared_new_points = SharedPoints(test_points.shape[0])
        self.shared_new_gaussians = SharedGaussians(test_points.shape[0])
        self.shared_target_gaussians = SharedTargetPoints(10000000)
        self.end_of_dataset = torch.zeros((1)).int()
        self.is_tracking_keyframe_shared = torch.zeros((1)).int()
        self.is_mapping_keyframe_shared = torch.zeros((1)).int()
        self.target_gaussians_ready = torch.zeros((1)).int()
        self.new_points_ready = torch.zeros((1)).int()
        self.final_pose = torch.zeros((num_final_poses,4,4)).float()
        self.demo = torch.zeros((1)).int()
        self.is_mapping_process_started = torch.zeros((1)).int()
        self.iter_shared = torch.zeros((1)).int()
        
        self.shared_cam.share_memory()
        self.shared_new_points.share_memory()
        self.shared_new_gaussians.share_memory()
        self.shared_target_gaussians.share_memory()
        self.end_of_dataset.share_memory_()
        self.is_tracking_keyframe_shared.share_memory_()
        self.is_mapping_keyframe_shared.share_memory_()
        self.target_gaussians_ready.share_memory_()
        self.new_points_ready.share_memory_()
        self.final_pose.share_memory_()
        self.demo.share_memory_()
        self.is_mapping_process_started.share_memory_()
        self.iter_shared.share_memory_()
        
        self.demo[0] = args.demo
        self.mapper = Mapper(self)
        self.tracker = Tracker(self)

    def tracking(self, rank):
        self.tracker.run()
    
    def mapping(self, rank):
        self.mapper.run()

    def run(self):
        processes = []
        for rank in range(2):
            if rank == 0:
                p = mp.Process(target=self.tracking, args=(rank, ))
            elif rank == 1:
                p = mp.Process(target=self.mapping, args=(rank, )) 
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def get_dataset(self, dataset_type, dataset_folder):
        if 'tartanair' in dataset_type.lower():
            dataset = TartanAirLoader(dataset_folder)
        elif 'tum' in dataset_type.lower():
            dataset = TUMLoader(dataset_folder)
        elif 'scannet' in dataset_type.lower():
            dataset = ScannetLoader(dataset_folder)
        elif 'replica' in dataset_type.lower():
            dataset = ReplicaLoader(dataset_folder)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        return dataset

    def get_test_image(self):
        return self.dataset.read_current_rgbd()

    def run_viewer(self, lower_speed=True):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            if time.time()-self.last_t < 1/self.viewer_fps and lower_speed:
                break
            try:
                net_image_bytes = None
                custom_cam, do_training, self.pipe.convert_SHs_python, self.pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                    # net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render_depth"]
                    # net_image = torch.concat([net_image,net_image,net_image], dim=0)
                    # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=7.0) * 50).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path) 
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def set_downsample_filter( self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * torch.arange(0,int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0,self.W,sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0,self.H), torch.arange(0,self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]
        
        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx # * z_values
        y_pre = (v-self.cy)/self.fy # * z_values
        
        return pick_idxs, x_pre, y_pre

    def downsample_and_make_pointcloud(self, depth_img, rgb_img):
        
        colors = torch.from_numpy(rgb_img).reshape(-1,3).float()[self.downsample_idxs]/255
        z_values = torch.from_numpy(depth_img.astype(np.float32)).flatten()[self.downsample_idxs]/self.depth_scale
        filter = torch.where((z_values!=0)&(z_values<=self.depth_trunc))
        # print(z_values[filter].min())
        # Trackable gaussians (will be used in tracking)
        z_values = z_values
        x = self.x_pre * z_values
        y = self.y_pre * z_values
        points = torch.stack([x,y,z_values], dim=-1)
        colors = colors
        
        # untrackable gaussians (won't be used in tracking, but will be used in 3DGS)
        
        return points.numpy(), colors.numpy(), z_values.numpy(), filter[0].numpy()

if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_path / output_path / verbose")
    parser.add_argument("--dataset_path", help="dataset path", default="dataset/Replica/room0")
    parser.add_argument("--config", help="caminfo", default="configs/Replica/caminfo.txt")
    parser.add_argument("--output_path", help="output path", default="output/room0")
    parser.add_argument("--keyframe_th", default=0.7)
    parser.add_argument("--knn_maxd", default=99999.0)
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--demo", action='store_true', default=False)
    parser.add_argument("--overlapped_th", default=5e-4)
    parser.add_argument("--max_correspondence_distance", default=0.02)
    parser.add_argument("--trackable_opacity_th", default=0.05)
    parser.add_argument("--overlapped_th2", default=5e-5)
    parser.add_argument("--downsample_rate", default=10)
    parser.add_argument("--test", default=None)
    parser.add_argument("--save_results", action='store_true', default=None)
    parser.add_argument("--rerun_viewer", action="store_true", default=False)
    args = parser.parse_args()

    gs_icp_slam = GS_ICP_SLAM(args)
    # gs_icp_slam.SLAM(1)
    gs_icp_slam.run()
