import os
import copy
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import rerun as rr
import torch
import torch.multiprocessing as mp
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

sys.path.append(os.path.dirname(__file__))

from arguments import SLAMParameters
from dataloader import (
    DataLoaderBase,
    TartanAirLoader,
    TUMLoader,
    ScannetLoader,
    ReplicaLoader
)
from gaussian_renderer import render, render_3, network_gui
from scene import GaussianModel
from utils.traj_utils import TrajManager
from utils.loss_utils import l1_loss, ssim
from utils.writers_image import ImageWriter, ImageWriterConfig
from utils.writers_pose import (
    TumTrajectoryWriter, TumTrajectoryWriterConfig,
    NumpyTrajectoryWriter, NumpyTrajectoryWriterConfig
)


class Pipe():
    def __init__(self, convert_SHs_python, compute_cov3D_python, debug):
        self.convert_SHs_python = convert_SHs_python
        self.compute_cov3D_python = compute_cov3D_python
        self.debug = debug


class Mapper(SLAMParameters):
    def __init__(self, slam):
        super().__init__()
        self.dataset_path = slam.dataset_path
        self.output_path = slam.output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.verbose = slam.verbose
        self.keyframe_th = float(slam.keyframe_th)
        self.trackable_opacity_th = slam.trackable_opacity_th
        self.save_results = slam.save_results
        self.rerun_viewer = slam.rerun_viewer
        self.iter_shared = slam.iter_shared

        self.W = slam.W
        self.H = slam.H
        self.fx = slam.fx
        self.fy = slam.fy
        self.cx = slam.cx
        self.cy = slam.cy
        self.depth_scale = slam.depth_scale
        self.depth_trunc = slam.depth_trunc
        self.cam_intrinsic = np.array([[self.fx, 0., self.cx],
                                       [0., self.fy, self.cy],
                                       [0., 0., 1]])

        self.downsample_rate = slam.downsample_rate
        self.viewer_fps = slam.viewer_fps
        self.keyframe_freq = slam.keyframe_freq

        # Camera poses
        # self.trajmanager = TrajManager(slam.config, self.dataset_path)
        # self.poses = [self.trajmanager.gt_poses[0]]
        # print(self.poses)
        self.poses = [np.array(slam.dataset[0][0])]
        # Keyframes(added to map gaussians)
        self.keyframe_idxs = []
        self.last_t = time.time()
        self.iteration_images = 0
        self.end_trigger = False
        self.covisible_keyframes = []
        self.new_target_trigger = False
        self.start_trigger = False
        self.if_mapping_keyframe = False
        self.cam_t = []
        self.cam_R = []
        self.points_cat = []
        self.colors_cat = []
        self.rots_cat = []
        self.scales_cat = []
        self.trackable_mask = []
        self.from_last_tracking_keyframe = 0
        self.from_last_mapping_keyframe = 0
        self.scene_extent = 2.5
        self.dataset_type = slam.config
        if self.dataset_type == "replica":
            self.prune_th = 2.5
        else:
            self.prune_th = 10.0

        self.downsample_idxs, self.x_pre, self.y_pre = self.set_downsample_filter(
            self.downsample_rate)

        self.gaussians = GaussianModel(self.sh_degree)
        self.pipe = Pipe(self.convert_SHs_python,
                         self.compute_cov3D_python, self.debug)
        self.bg_color = [1, 1, 1] if self.white_background else [0, 0, 0]
        self.background = torch.tensor(
            self.bg_color, dtype=torch.float32, device="cuda")
        self.train_iter = 0
        self.mapping_cams = []
        self.mapping_losses = []
        self.new_keyframes = []
        self.gaussian_keyframe_idxs = []

        self.shared_cam = slam.shared_cam
        self.shared_new_points = slam.shared_new_points
        self.shared_new_gaussians = slam.shared_new_gaussians
        self.shared_target_gaussians = slam.shared_target_gaussians
        self.end_of_dataset = slam.end_of_dataset
        self.is_tracking_keyframe_shared = slam.is_tracking_keyframe_shared
        self.is_mapping_keyframe_shared = slam.is_mapping_keyframe_shared
        self.target_gaussians_ready = slam.target_gaussians_ready
        self.final_pose = slam.final_pose
        self.demo = slam.demo
        self.is_mapping_process_started = slam.is_mapping_process_started

        self.image_writer: ImageWriter = ImageWriterConfig(
            workdir=Path(self.output_path) / "renders",
            enabled=True,
        ).setup()
        self.pose_writer_tum: TumTrajectoryWriter = TumTrajectoryWriterConfig(
            filename=Path(self.output_path) / "traj_tum.txt"
        ).setup()
        self.pose_writer_numpy: NumpyTrajectoryWriter = NumpyTrajectoryWriterConfig(
            filename=Path(self.output_path) / "traj.npz"
        ).setup()

    def run(self):
        self.mapping()

    def mapping(self):
        t = torch.zeros((1, 1)).float().cuda()
        if self.verbose:
            network_gui.init("127.0.0.1", 6009)

        if self.rerun_viewer:
            rr.init("3dgsviewer")
            rr.connect()

        # Mapping Process is ready to receive first frame
        self.is_mapping_process_started[0] = 1

        # Wait for initial gaussians
        while not self.is_tracking_keyframe_shared[0]:
            time.sleep(1e-15)

        self.total_start_time_viewer = time.time()

        points, colors, rots, scales, z_values, trackable_filter = self.shared_new_gaussians.get_values()
        self.gaussians.create_from_pcd2_tensor(
            points, colors, rots, scales, z_values, trackable_filter)
        self.gaussians.spatial_lr_scale = self.scene_extent
        self.gaussians.training_setup(self)
        self.gaussians.update_learning_rate(1)
        self.gaussians.active_sh_degree = self.gaussians.max_sh_degree
        self.is_tracking_keyframe_shared[0] = 0

        if self.demo[0]:
            a = time.time()
            while (time.time()-a) < 30.:
                print(30.-(time.time()-a))
                self.run_viewer()
        self.demo[0] = 0

        newcam = copy.deepcopy(self.shared_cam)
        newcam.on_cuda()

        self.mapping_cams.append(newcam)
        self.keyframe_idxs.append(newcam.cam_idx[0])
        self.new_keyframes.append(len(self.mapping_cams)-1)

        new_keyframe = False
        while True:
            if self.end_of_dataset[0]:
                break

            if self.verbose:
                self.run_viewer()

            if self.is_tracking_keyframe_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, trackable_filter = self.shared_new_gaussians.get_values()

                # Add new gaussians to map gaussians
                self.gaussians.add_from_pcd2_tensor(
                    points, colors, rots, scales, z_values, trackable_filter)

                # Allocate new target points to shared memory
                target_points, target_rots, target_scales = self.gaussians.get_trackable_gaussians_tensor(
                    self.trackable_opacity_th)
                self.shared_target_gaussians.input_values(
                    target_points, target_rots, target_scales)
                self.target_gaussians_ready[0] = 1

                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()

                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.is_tracking_keyframe_shared[0] = 0

            elif self.is_mapping_keyframe_shared[0]:
                # get shared gaussians
                points, colors, rots, scales, z_values, _ = self.shared_new_gaussians.get_values()

                # Add new gaussians to map gaussians
                self.gaussians.add_from_pcd2_tensor(
                    points, colors, rots, scales, z_values, [])

                # Add new keyframe
                newcam = copy.deepcopy(self.shared_cam)
                newcam.on_cuda()
                self.mapping_cams.append(newcam)
                self.keyframe_idxs.append(newcam.cam_idx[0])
                self.new_keyframes.append(len(self.mapping_cams)-1)
                self.is_mapping_keyframe_shared[0] = 0

            if len(self.mapping_cams) > 0:

                # train once on new keyframe, and random
                if len(self.new_keyframes) > 0:
                    train_idx = self.new_keyframes.pop(0)
                    viewpoint_cam = self.mapping_cams[train_idx]
                    new_keyframe = True
                else:
                    train_idx = random.choice(range(len(self.mapping_cams)))
                    viewpoint_cam = self.mapping_cams[train_idx]

                if self.training_stage == 0:
                    gt_image = viewpoint_cam.original_image.cuda()
                    gt_depth_image = viewpoint_cam.original_depth_image.cuda()
                elif self.training_stage == 1:
                    gt_image = viewpoint_cam.rgb_level_1.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_1.cuda()
                elif self.training_stage == 2:
                    gt_image = viewpoint_cam.rgb_level_2.cuda()
                    gt_depth_image = viewpoint_cam.depth_level_2.cuda()

                self.training = True
                render_pkg = render_3(viewpoint_cam, self.gaussians, self.pipe,
                                      self.background, training_stage=self.training_stage)

                depth_image = render_pkg["render_depth"]
                image = render_pkg["render"]
                viewspace_point_tensor, visibility_filter, radii = render_pkg[
                    "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

                mask = (gt_depth_image > 0.)
                mask = mask.detach()
                # color_mask = torch.tile(mask, (3,1,1))
                gt_image = gt_image * mask

                # Loss
                Ll1_map, Ll1 = l1_loss(image, gt_image)
                L_ssim_map, L_ssim = ssim(image, gt_image)

                d_max = 10.
                Ll1_d_map, Ll1_d = l1_loss(
                    depth_image/d_max, gt_depth_image/d_max)

                loss_rgb = (1.0 - self.lambda_dssim) * Ll1 + \
                    self.lambda_dssim * (1.0 - L_ssim)
                loss_d = Ll1_d

                loss = loss_rgb + 0.1*loss_d

                loss.backward()
                with torch.no_grad():
                    if self.train_iter % 200 == 0:  # 200
                        self.gaussians.prune_large_and_transparent(
                            0.005, self.prune_th)

                    self.gaussians.optimizer.step()
                    self.gaussians.optimizer.zero_grad(set_to_none=True)

                    if new_keyframe and self.rerun_viewer:
                        current_i = copy.deepcopy(self.iter_shared[0])
                        rgb_np = image.cpu().numpy().transpose(1, 2, 0)
                        rgb_np = np.clip(rgb_np, 0., 1.0) * 255
                        # rr.set_time_sequence("step", current_i)
                        rr.set_time_seconds(
                            "log_time", time.time() - self.total_start_time_viewer)
                        rr.log("rendered_rgb", rr.Image(rgb_np))
                        new_keyframe = False

                self.training = False
                self.train_iter += 1
                # torch.cuda.empty_cache()
        if self.verbose:
            while True:
                self.run_viewer(False)

        # End of data
        if self.save_results and not self.rerun_viewer:
            self.gaussians.save_ply(os.path.join(
                self.output_path, "scene.ply"))

        self.calc_2d_metric()

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
                    net_image = render(
                        custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(
                        net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                    # net_image = render(custom_cam, self.gaussians, self.pipe, self.background, scaling_modifer)["render_depth"]
                    # net_image = torch.concat([net_image,net_image,net_image], dim=0)
                    # net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=7.0) * 50).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                self.last_t = time.time()
                network_gui.send(net_image_bytes, self.dataset_path)
                if do_training and (not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

    def set_downsample_filter(self, downsample_scale):
        # Get sampling idxs
        sample_interval = downsample_scale
        h_val = sample_interval * \
            torch.arange(0, int(self.H/sample_interval)+1)
        h_val = h_val-1
        h_val[0] = 0
        h_val = h_val*self.W
        a, b = torch.meshgrid(h_val, torch.arange(0, self.W, sample_interval))
        # For tensor indexing, we need tuple
        pick_idxs = ((a+b).flatten(),)
        # Get u, v values
        v, u = torch.meshgrid(torch.arange(0, self.H), torch.arange(0, self.W))
        u = u.flatten()[pick_idxs]
        v = v.flatten()[pick_idxs]

        # Calculate xy values, not multiplied with z_values
        x_pre = (u-self.cx)/self.fx  # * z_values
        y_pre = (v-self.cy)/self.fy  # * z_values

        return pick_idxs, x_pre, y_pre

    def get_dataset(self):

        dataset_type = self.dataset_type
        dataset_folder = self.dataset_path

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

    def calc_2d_metric(self):
        psnrs = []
        ssims = []
        lpips = []

        cal_lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True).to("cuda")
        dataset = self.get_dataset()
        final_poses = self.final_pose
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                gt_depth_ = []
                cam = self.mapping_cams[0]
                c2w = final_poses[i]

                dataset.set_curr_index(i)
                gt_rgb, gt_depth = dataset.read_current_rgbd()
                gt_depth = gt_depth.astype(np.float32)

                gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_RGB2BGR)
                gt_rgb = gt_rgb/255
                gt_rgb_ = torch.from_numpy(
                    gt_rgb).float().cuda().permute(2, 0, 1)

                gt_depth_ = torch.from_numpy(
                    gt_depth).float().cuda().unsqueeze(0)

                w2c = np.linalg.inv(c2w)
                # rendered
                R = w2c[:3, :3].transpose()
                T = w2c[:3, 3]

                cam.R = torch.tensor(R)
                cam.t = torch.tensor(T)
                cam.image_width = gt_rgb_.shape[2]
                cam.image_height = gt_rgb_.shape[1]

                cam.update_matrix()
                # rendered rgb
                render_pkg = render(cam, self.gaussians, self.pipe, self.background)
                ours_rgb_ = render_pkg["render"]
                ours_depth_ = render_pkg["render_depth"]
                ours_rgb_ = torch.clamp(ours_rgb_, 0., 1.).cuda()

                valid_depth_mask_ = (gt_depth_ > 0)

                gt_rgb_ = gt_rgb_ * valid_depth_mask_
                ours_rgb_ = ours_rgb_ * valid_depth_mask_

                square_error = (gt_rgb_-ours_rgb_)**2
                mse_error = torch.mean(torch.mean(square_error, axis=2))
                psnr = mse2psnr(mse_error)

                psnrs += [psnr.detach().cpu()]
                _, ssim_error = ssim(ours_rgb_, gt_rgb_)
                ssims += [ssim_error.detach().cpu()]
                lpips_value = cal_lpips(
                    gt_rgb_.unsqueeze(0), ours_rgb_.unsqueeze(0))
                lpips += [lpips_value.detach().cpu()]

                # if self.save_results and (i % 100 == 0 or i == len(dataset)-1):
                if self.save_results:
                    ours_rgb_npy = np.asarray(
                        ours_rgb_.detach().cpu()).squeeze().transpose((1, 2, 0))


                    self.image_writer.write_image(
                        image_name=f"rgb_{i:04}",
                        data=ours_rgb_,
                    )
                    self.image_writer.write_image(
                        image_name=f"depth_{i:04}",
                        data=ours_depth_,
                    )
                    self.pose_writer_tum.write(
                        timestamp=i,
                        pose=c2w,
                    )
                    self.pose_writer_numpy.write(
                        timestamp=i,
                        pose=c2w,
                    )

                torch.cuda.empty_cache()

            psnrs = np.array(psnrs)
            ssims = np.array(ssims)
            lpips = np.array(lpips)

            print(
                f"PSNR: {psnrs.mean():.2f}\nSSIM: {ssims.mean():.3f}\nLPIPS: {lpips.mean():.3f}")


def mse2psnr(x):
    return -10.*torch.log(x)/torch.log(torch.tensor(10.))
