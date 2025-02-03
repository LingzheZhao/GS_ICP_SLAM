import os
import numpy as np
import cv2
from evo.tools import file_interface
from spatialmath import *
import plotly.graph_objects as go
from icecream import ic
from spatialmath.base import trnorm


class DataLoaderBase():
    def __init__(self, dataset_folder):
        self.dataset_folder = self._fix_path(
            os.path.expanduser(dataset_folder))
        self.rgb_folder = None
        self.depth_folder = None
        self.stereo_folders = None
        self.gt_filename = 'pose_left.txt'
        self.odom_filename = 'pose_left.txt'
        self.curr_index = 0
        self.index_interval = 1
        self.start_index = 0
        self.end_index = -1
        self.camera = [0, 0, 0, 0]  # fx, fy, cx, cy
        self.image_size = (0, 0)  # width, height
        self.depth_scale = 1.0
        self.depth_trunc = 10.

    def read_current_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def read_current_stereo(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    def read_current_ground_truth(self) -> SE3:
        raise NotImplementedError()

    def read_current_odometry(self) -> SE3:
        raise NotImplementedError()

    def load_ground_truth(self) -> None:
        raise NotImplementedError()

    def load_odometry(self) -> None:
        raise NotImplementedError()

    def set_odometry(self, traj) -> None:
        raise NotImplementedError()

    def get_total_number(self) -> int:
        '''
        count number of frames according to number of files in color folder
        '''
        if self.rgb_folder is not None:
            dir_path = self.dataset_folder + self.rgb_folder
        elif self.stereo_folders is not None:
            dir_path = self.dataset_folder + self.stereo_folders[0]
        return len([entry for entry in os.listdir(dir_path)
                    if os.path.isfile(os.path.join(dir_path, entry))])

    def _load_traj(self, traj_type, traj_filename, ignore_timestamps=False, add_timestamps=False) -> SE3:
        '''
        load trajectory from file
        @param traj_type: ['kitti', 'tum', 'euroc']
        @param ignore_timestamps: if True, ignore the timestamps in the first column
        @param add_timestamps: if True, add timestamps to the first column
        @return: SE3
        '''
        function_dict = {
            'kitti': file_interface.read_kitti_poses_file,
            'tum': file_interface.read_tum_trajectory_file,
            'euroc': file_interface.read_euroc_csv_trajectory
        }
        file_path = self.dataset_folder + traj_filename
        if ignore_timestamps or add_timestamps:
            if not os.path.exists('tmp'):
                os.makedirs('tmp')
            tmp_file_path = 'tmp/tmp_' + traj_filename
            with open(file_path, 'r') as raw_file:
                lines = raw_file.readlines()
            with open(tmp_file_path, 'w') as output_file:
                for i, line in enumerate(lines):
                    if ignore_timestamps:
                        new_line = ' '.join(line.split(' ')[1:])
                    elif add_timestamps:
                        new_line = str(i)+' '+line
                    output_file.write(new_line)
            file_path = tmp_file_path
        traj_evo = function_dict[traj_type](file_path)
        # convert PoseTrajectory3D to SE3
        traj = SE3(traj_evo.poses_se3)
        traj.timestamps = traj_evo.timestamps
        # remove tmp file
        if ignore_timestamps or add_timestamps:
            os.remove(tmp_file_path)
        return traj

    def _load_array_from_txt(self, file_path, format='4x4'):
        '''
        format: '4x4' or 'N x16'
        '''
        if format == '4x4':
            with open(file_path, 'r') as f:
                lines = f.readlines()
            return np.array([[float(x) for x in line.split()] for line in lines])
        elif format == 'Nx16':
            with open(file_path, 'r') as f:
                lines = f.readlines()
            return np.array([[float(x) for x in line.split()] for line in lines]).reshape(-1, 4, 4)

    def _zeros(self, str_length, num) -> str:
        '''
        @param str_length: length of the string
        @param num: number of zeros
        @return: a string with leading zeros
        '''
        return '0' * (str_length - len(str(num))) + str(num)

    def _fix_path(self, path) -> str:
        '''
        add a / at the end of the path if it is not there
        '''
        return path if path[-1] == '/' else path+'/'

    def get_curr_index(self) -> int:
        return self.curr_index

    def set_curr_index(self, index) -> None:
        self.curr_index = index

    def load_next_frame(self) -> bool:
        '''
        @return: True if there are still frames to load
        '''
        self.curr_index += self.index_interval
        if self.end_index > 0:
            return self.curr_index < self.end_index
        else:
            return True

    def add_noise(self, traj, mean_sigma=[2e-4, 2e-4], sigma=[1e-3, 1e-3], seed=None, start=0) -> SE3:
        '''
        add noise to the trajectory
        @param traj: SE3
        @param mean_sigma: standard deviation of the mean of the noise [translation, rotation]
        @param sigma: standard deviation of the noise [translation, rotation]
        @param seed: random seed
        @return: SE3
        '''
        if seed is None:
            seed = np.random.randint(0, 100000)
            print(f'Adding noise, seed={seed}')
        np.random.seed(seed)
        new_traj = SE3(traj)
        noise = SE3()
        noise_t_bias = SE3.Trans(np.random.normal(0, mean_sigma[0], 3))
        noise_r_bias = SE3.RPY(*np.random.normal(0, mean_sigma[1], 3))
        for i in range(start+1, len(traj)):
            noise_t_delta = SE3.Trans(
                np.random.normal(0, sigma[0], 3)) * noise_t_bias
            noise_r_delta = SE3.RPY(
                *np.random.normal(0, sigma[1], 3)) * noise_r_bias
            noise = noise_r_delta * noise_t_delta * noise
            new_pose = noise * new_traj[i]
            new_traj[i] = new_pose
        return new_traj

    def set_range(self, start, end=-1, interval=1) -> None:
        n_total = self.get_total_number()
        start = min(max(0, start), n_total-1)
        if end < 0:
            end = n_total
        end = n_total if end < 0 else end
        end = min(end, n_total)
        self.curr_index = start-interval
        self.start_index = start
        self.end_index = end
        self.index_interval = interval
        return start, end, interval

    def __len__(self) -> int:
        start = self.start_index
        int = self.index_interval
        end = self.end_index
        end = self.get_total_number() if end < 0 else end
        return (end - start - 1) // int + 1

    def __getitem__(self, index):
        self.set_curr_index(index)
        pose_gt = self.read_current_ground_truth()
        rgb, depth = self.read_current_rgbd()
        return pose_gt, rgb, depth

    def __iter__(self):
        return self

    def __next__(self):
        if self.load_next_frame():
            return self.curr_index
        else:
            raise StopIteration()

    def __str__(self) -> str:
        return f'<{self.__class__.__name__}, path: {self.dataset_folder}, range: ({self.start_index}, {self.end_index}, {self.index_interval})>'

    def __repr__(self) -> str:
        return self.__str__()


class TartanAirLoader(DataLoaderBase):
    def __init__(self, dataset_folder, depth_folder='depth_left/',
                 stereo_folders_left='image_left/', stereo_folders_right='image_right/'):
        super().__init__(dataset_folder)
        self.depth_folder = self._fix_path(depth_folder)
        self.stereo_folders = [
            self._fix_path(stereo_folders_left),
            self._fix_path(stereo_folders_right)
        ]
        self.gt_filename = 'pose_left.txt'
        self.odom_filename = 'pose_left.txt'
        self.camera = [320, 320, 320, 240]  # fx, fy, cx, cy
        self.image_size = (640, 480)  # width, height
        self.depth_scale = 1.0
        self.depth_trunc = 40.
        self.load_ground_truth()

    def read_current_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        index_str = super()._zeros(6, self.curr_index)
        left_color = cv2.imread(
            f'{self.dataset_folder}{self.stereo_folders[0]}{index_str}_left.png')
        left_color = cv2.cvtColor(left_color, cv2.COLOR_BGR2RGB)
        left_depth = np.load(
            f'{self.dataset_folder}{self.depth_folder}{index_str}_left_depth.npy')
        return (left_color, left_depth)

    def read_current_stereo(self) -> tuple[np.ndarray, np.ndarray]:
        index_str = self._zeros(6, self.curr_index)
        left_color = cv2.imread(
            f'{self.dataset_folder}{self.stereo_folders[0]}{index_str}_left.png')
        right_color = cv2.imread(
            f'{self.dataset_folder}{self.stereo_folders[1]}{index_str}_right.png')
        return (left_color, right_color)

    def read_current_ground_truth(self) -> SE3:
        return self.gt[self.curr_index]

    def read_current_odometry(self) -> SE3:
        return self.odom[self.curr_index]

    def load_ground_truth(self) -> None:
        poses = self._load_traj('tum', 'pose_left.txt', add_timestamps=True)
        T = np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        T_inv = np.linalg.inv(T)
        gt = []
        for pose in poses.data:
            # gt.append(SE3(pose))
            gt.append(SE3(T@pose@T_inv))
        self.gt = SE3(gt)

    def set_ground_truth(self, traj) -> None:
        self.gt = traj

    # def load_odometry(self, traj=None) -> None:
    #     self.odom = self._load_traj(
    #         'tum', 'pose_left.txt', add_timestamps=True)

    # def set_odometry(self, traj) -> None:
    #     self.odom = traj


class TUMLoader(DataLoaderBase):
    def __init__(self, dataset_folder, depth_folder='depth/', rgb_folder='rgb/'):
        super().__init__(dataset_folder)
        self.depth_folder = self._fix_path(depth_folder)
        self.rgb_folder = self._fix_path(rgb_folder)
        self.gt_filename = 'groundtruth.txt'
        self.camera = [525.0, 525.0, 319.5, 239.5]  # fx, fy, cx, cy
        self.image_size = (640, 480)  # width, height
        self.depth_scale = 1.0
        self.depth_trunc = 3.
        # load file names
        self.rgb_files = []
        for root, dirs, files in os.walk(self.dataset_folder+self.rgb_folder):
            for file in files:
                if file.endswith('.png'):
                    self.rgb_files.append(file[:-4])
        self.depth_files = []
        for root, dirs, files in os.walk(self.dataset_folder+self.depth_folder):
            for file in files:
                if file.endswith('.png'):
                    self.depth_files.append(file[:-4])
        self.rgb_files = sorted(self.rgb_files)
        self.depth_files = sorted(self.depth_files)
        rgb_timestamp = np.array([float(x) for x in self.rgb_files])
        depth_timestamp = np.array([float(x) for x in self.depth_files])
        time_diff = rgb_timestamp.reshape(
            (-1, 1)) - depth_timestamp.reshape((1, -1))
        self.associations = []  # (rgb, depth)
        if len(self.rgb_files) > len(self.depth_files):
            min_diff_index = np.argmin(np.abs(time_diff), axis=0)
            self.associations = [(self.rgb_files[min_diff_index[i]], self.depth_files[i])
                                 for i in range(len(self.depth_files))]
        else:
            min_diff_index = np.argmin(np.abs(time_diff), axis=1)
            self.associations = [(self.rgb_files[i], self.depth_files[min_diff_index[i]])
                                 for i in range(len(self.rgb_files))]
        self.load_ground_truth()

    def get_total_number(self) -> int:
        '''
        count number of frames according to number of files in color folder
        '''
        return len(self.associations)

    def read_current_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        association = self.associations[self.curr_index]
        rgb = cv2.imread(
            f'{self.dataset_folder}{self.rgb_folder}{association[0]}.png')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(
            f'{self.dataset_folder}{self.depth_folder}{association[1]}.png', cv2.IMREAD_ANYDEPTH)/5000.0
        return (rgb, depth)

    def read_current_ground_truth(self) -> SE3:
        return self.gt[self.curr_index]

    def load_ground_truth(self) -> None:
        poses = self._load_traj('tum', self.gt_filename)

        # gsplat
        T = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

        T_inv = np.linalg.inv(T)

        # associate gt to self.associations
        rgb_timestamp = np.array([float(x[0]) for x in self.associations])
        traj_timestamp = np.array(poses.timestamps)
        time_diff = rgb_timestamp.reshape(
            (-1, 1)) - traj_timestamp.reshape((1, -1))
        min_diff_index = np.argmin(np.abs(time_diff), axis=1)
        associations = [(rgb_timestamp[i], traj_timestamp[min_diff_index[i]])
                        for i in range(len(rgb_timestamp))]
        # print(associations)
        gt = [poses.data[min_diff_index[i]] for i in range(len(rgb_timestamp))]
        # gt = [pose for pose in gt]
        # gt = [T@pose@T_inv for pose in gt]
        self.gt = SE3(gt)

    def set_ground_truth(self, traj) -> None:
        self.gt = traj


class KittiLoader(DataLoaderBase):
    def __init__(self, dataset_folder, depth_folder='depth/', rgb_folder='rgb/'):
        super().__init__(dataset_folder)
        self.depth_folder = self._fix_path(depth_folder)
        self.rgb_folder = self._fix_path(rgb_folder)
        self.gt_filename = 'groundtruth.txt'
        self.camera = [525.0, 525.0, 319.5, 239.5]


class ScannetLoader(DataLoaderBase):
    def __init__(self, dataset_folder, depth_folder='depth/', rgb_folder='color/'):
        super().__init__(dataset_folder)
        self.depth_folder = self._fix_path(depth_folder)
        self.rgb_folder = self._fix_path(rgb_folder)
        self.pose_folder = self._fix_path('pose/')
        self.intrinsic_folder = self._fix_path('intrinsic/')
        self.camera, self.image_size = self.load_intrinsics()
        self.depth_scale = 1.0
        self.depth_trunc = 10.
        self.load_ground_truth()

    def load_intrinsics(self) -> tuple[list, tuple]:
        intrinsics_file = f'{self.dataset_folder}{self.intrinsic_folder}intrinsic_color.txt'
        intrinsics = self._load_array_from_txt(intrinsics_file)
        camera = [intrinsics[0][0], intrinsics[1][1],
                  intrinsics[0][2], intrinsics[1][2]]  # fx, fy, cx, cy

        first_rgb = cv2.imread(f'{self.dataset_folder}{self.rgb_folder}0.jpg')
        image_size = (first_rgb.shape[1], first_rgb.shape[0])
        return camera, image_size

    def read_current_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        index_str = str(self.curr_index)
        rgb = cv2.imread(
            f'{self.dataset_folder}{self.rgb_folder}{index_str}.jpg')
        depth = cv2.imread(
            f'{self.dataset_folder}{self.depth_folder}{index_str}.png', cv2.IMREAD_ANYDEPTH) / 1000.0
        depth = cv2.resize(depth, self.image_size,
                           interpolation=cv2.INTER_NEAREST)
        return rgb, depth

    def read_current_ground_truth(self) -> SE3:
        return self.gt[self.curr_index]

    def load_ground_truth(self) -> None:
        gt = []
        for i in range(len(self)):
            pose_file = f'{self.dataset_folder}{self.pose_folder}{i}.txt'
            pose = self._load_array_from_txt(pose_file)
            if pose[3][0] != 0:
                pose = last_pose
            gt.append(SE3(trnorm(pose)))
            last_pose = pose
        self.gt = SE3(gt)


class ReplicaLoader(DataLoaderBase):
    def __init__(self, dataset_folder):
        super().__init__(dataset_folder)
        self.depth_folder = self._fix_path('results/')
        self.rgb_folder = self._fix_path('results/')
        self.gt_filename = 'traj.txt'
        self.camera = [600.0, 600.0, 599.5, 339.5]  # fx, fy, cx, cy
        self.image_size = (1200, 680)  # width, height
        self.depth_scale = 1.0
        self.depth_trunc = 12.
        self.load_ground_truth()

    def read_current_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        index_str = super()._zeros(6, self.curr_index)
        rgb = cv2.imread(
            f'{self.dataset_folder}{self.rgb_folder}frame{index_str}.jpg')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(
            f'{self.dataset_folder}{self.depth_folder}depth{index_str}.png', cv2.IMREAD_ANYDEPTH)/6553.5
        return (rgb, depth)

    def get_total_number(self) -> int:
        dir_path = self.dataset_folder + self.rgb_folder
        return int(len([entry for entry in os.listdir(dir_path)
                        if os.path.isfile(os.path.join(dir_path, entry))])/2)

    def read_current_ground_truth(self) -> SE3:
        return self.gt[self.curr_index]

    def load_ground_truth(self) -> None:
        poses = self._load_array_from_txt(
            f'{self.dataset_folder}{self.gt_filename}', format='Nx16')
        gt = []
        for pose in poses:
            gt.append(SE3(trnorm(pose)))
        self.gt = SE3(gt)


def load_dataset(cfg) -> DataLoaderBase:
    cfg = cfg.dataset
    if cfg.type == 'tartanair':
        dataset = TartanAirLoader(cfg.folder)
    elif cfg.type == 'tum':
        dataset = TUMLoader(cfg.folder)
    elif cfg.type == 'scannet':
        dataset = ScannetLoader(cfg.folder)
    elif cfg.type == 'replica':
        dataset = ReplicaLoader(cfg.folder)
    else:
        raise NotImplementedError

    dataset.set_range(cfg.start_index, cfg.end_index, cfg.frame_interval)

    return dataset
