import yaml
from dataclasses import dataclass, field
import numpy as np
from os.path import expanduser, expandvars
from robotdatapy.data import PoseData, ImgData
from robotdatapy.transform import T_FLURDF, T_RDFFLU
from functools import cached_property
from typing import Tuple
import os

def find_transformation(bag_path, param_dict) -> np.array:
    """
    Converts a transform parameter dictionary into a transformation matrix.

    Returns:
        np.array: Transformation matrix.
    """
    if param_dict['input_type'] == 'tf':
        bag_path = expandvars_recursive(bag_path)
        T = PoseData.any_static_tf_from_bag(
            expandvars_recursive(bag_path), 
            expandvars_recursive(param_dict['parent']), 
            expandvars_recursive(param_dict['child'])
        )
        if 'inv' in param_dict.keys() and param_dict['inv']:
            T = np.linalg.inv(T)
        return T
    elif param_dict['input_type'] == 'matrix':
        return np.array(param_dict['matrix']).reshape((4, 4))
    elif param_dict['input_type'] == 'string':
        if param_dict['string'] == 'T_FLURDF':
            return T_FLURDF
        elif param_dict['string'] == 'T_RDFFLU':
            return T_RDFFLU
        else:
            raise ValueError("Invalid string.")
    else:
        raise ValueError("Invalid input type.")

def expandvars_recursive(path):
    """Recursively expands environment variables in the given path."""
    while True:
        expanded_path = expandvars(path)
        if expanded_path == path:
            return expanduser(expanded_path)
        path = expanded_path

@dataclass
class ImgDataParams:
    
    path: str
    topic: str
    camera_info_topic: str
    time_tol: float = float('inf')
    compressed: bool = True
    compressed_rvl: bool = False
    
    @classmethod
    def from_dict(cls, params_dict: dict):
        return cls(**params_dict)
    
@dataclass
class DepthDataParams(ImgDataParams):
    depth_scale: float = 1000.0
    max_depth: float = None
    bilateral_smooth_depth: list = None


@dataclass
class PoseDataParams:
    
    params_dict: dict
    T_odom_camera: np.array
    
    @classmethod
    def from_dict(cls, params_dict: dict):
        params_dict_subset = {k: v for k, v in params_dict.items() 
                       if k != 'T_odom_camera'}
        T_odom_camera_dict = params_dict['T_odom_camera'] \
            if 'T_odom_camera' in params_dict else None
        T_odom_camera = find_transformation(params_dict_subset['path'], T_odom_camera_dict) if T_odom_camera_dict is not None else np.eye(4)
        params_dict['T_postmultiply'] = T_odom_camera
        return cls(params_dict=params_dict_subset,
                   T_odom_camera=T_odom_camera)
        
    def load_camera_pose_data(self, extra_key_vals: dict) -> PoseData:
        params_dict = {k: v for k, v in self.params_dict.items()}
        for k, v in extra_key_vals.items():
            params_dict[k] = v
            
        for k, v in params_dict.items():
            if type(v) == str:
                params_dict[k] = expandvars_recursive(v)

        pose_data = PoseData.from_dict(params_dict)
        return pose_data


@dataclass
class RaftArgs:
    small: bool = False
    mixed_precision: bool = False
    alternate_corr: bool = False

    @classmethod
    def from_dict(cls, args_dict):
        return cls(**args_dict)
    
    def __iter__(self):
        return iter(self.__dict__.items())
    
@dataclass
class RaftParams:
    path: str
    model: str
    raft_args: RaftArgs
    iters: int = 12
    device: str = 'cuda'

    def expand_vars(self):
        self.path = expandvars_recursive(self.path)
        self.model = expandvars_recursive(self.model)

    @classmethod
    def from_dict(cls, params_dict):
        params_dict['raft_args'] = RaftArgs.from_dict(params_dict['raft_args']) if 'raft_args' in params_dict else RaftArgs()
        return cls(**params_dict)
    
@dataclass
class TrackingParams:
    min_vel_threshold: float
    vel_threshold_gain: float
    max_merge_dist: float
    gaussian_smoothing: bool = False
    gaussian_kernel_size: int = 0
    post_processing: list = field(default_factory=list)
    min_3d_std_dev: float = None
    max_3d_std_dev: float = None
    min_consecutive_frames: int = 1

    @classmethod
    def from_dict(cls, params_dict):
        return cls(**params_dict)

@dataclass
class ProcessingParams:
    # 3D Flows
    use_raft_flows_3d: bool = True
    use_geometric_flows_3d: bool = True
    use_residual_flow_3d: bool = True
    
    # 2D Flows
    use_raft_flows_2d: bool = True
    use_gflow_2d: bool = True
    use_residual_2d: bool = True
    
    # Other Data
    use_depth_data: bool = True
    use_image_data: bool = True

    @classmethod
    def from_dict(cls, params_dict):
        return cls(**params_dict)

@dataclass
class VizParams:
    viz_video: bool
    viz_dynamic_object_masks: bool
    vid_dims: list
    viz_flag_names: list
    viz_flags: list
    viz_max_residual_magnitude: float

    def __post_init__(self):
        self.viz_flags = np.array(self.viz_flags)
        self.viz_dynamic_object_masks = self.viz_video and self.viz_name('image') and self.viz_dynamic_object_masks

    @property
    def viz_residual(self):
        return self.viz_name('residual')
    
    def viz_name(self, name):
        return np.any(self.viz_flags == self.viz_flag_names.index(name))

    @classmethod
    def from_dict(cls, params_dict):
        return cls(**params_dict)

@dataclass
class Params:

    use_3d: bool
    img_data_params: ImgDataParams
    depth_data_params: DepthDataParams
    pose_data_params: PoseDataParams
    raft_params: RaftParams
    processing_params: ProcessingParams
    tracking_params: TrackingParams
    viz_params: VizParams
    time_params: dict
    device: str
    batch_size: int
    original_fps: int
    skip_frames: int
    output: str
    kmd_env: str = None
    robot: str = None
    raft_model: str = None

    def __post_init__(self):
        if self.kmd_env is not None:
            os.environ['KMD_ENV'] = self.kmd_env
        if self.robot is not None:
            os.environ['ROBOT'] = self.robot
        if self.raft_model is not None:
            os.environ['RAFT_MODEL'] = self.raft_model

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as fin:
            params = yaml.safe_load(fin)
        return cls(
            use_3d=params['use_3d'] if 'use_3d' in params else False,
            img_data_params=ImgDataParams.from_dict(params['img_data']),
            depth_data_params=DepthDataParams.from_dict(params['depth_data']),
            pose_data_params=PoseDataParams.from_dict(params['pose_data']),
            raft_params=RaftParams.from_dict(params['raft']),
            processing_params=ProcessingParams.from_dict(params['processing']),
            tracking_params=TrackingParams.from_dict(params['tracking']),
            viz_params=VizParams.from_dict(params['viz']),
            time_params=params['time'] if 'time' in params else None,
            device=params['device'] if 'device' in params else 'cuda',
            batch_size=params['batch_size'] if 'batch_size' in params else 24,
            original_fps=params['original_fps'] if 'original_fps' in params else 30,
            skip_frames=params['skip_frames'] if 'skip_frames' in params else 1,
            output=params['output'] if 'output' in params else 'output',
            kmd_env=params['kmd_env'] if 'kmd_env' in params else None,
            robot=params['robot'] if 'robot' in params else None,
            raft_model=params['raft_model'] if 'raft_model' in params else None,
        )
    
    @cached_property
    def time_range(self) -> Tuple[float, float]|None:
        return self._extract_time_range()
    
    def load_camera_pose_data(self) -> PoseData:
        extra_key_vals={'T_postmultiply': self.pose_data_params.T_odom_camera, 'interp': True}
        return self.pose_data_params.load_camera_pose_data(extra_key_vals)
    
    def load_img_data(self) -> ImgData:
        return self._load_img_data(color=True)
    
    def load_depth_data(self) -> ImgData:
        return self._load_img_data(color=False)

    def _extract_time_range(self) -> Tuple[float, float]:
        if self.time_params is not None:
            if 'relative' in self.time_params and self.time_params['relative']:
                topic_t0 = self.data_t0
                time_range = [topic_t0 + self.time_params['t0'], 
                                topic_t0 + self.time_params['tf']]
            else:
                time_range = [self.time_params['t0'], 
                                self.time_params['tf']]
        else:
            time_range = None
        return time_range
    
    def _load_img_data(self, color=True) -> ImgData:
        img_data_params = self.img_data_params if color else self.depth_data_params
        
        img_file_path = expandvars_recursive(img_data_params.path)
        img_data = ImgData.from_bag(
            path=img_file_path,
            topic=expandvars_recursive(img_data_params.topic),
            time_tol=img_data_params.time_tol,
            time_range=self.time_range,
            compressed=img_data_params.compressed,
            compressed_rvl=img_data_params.compressed_rvl,
        )
        img_data.extract_params(expandvars_recursive(img_data_params.camera_info_topic))
        return img_data
    
    @cached_property
    def data_t0(self) -> float:
        return ImgData.topic_t0(expandvars_recursive(self.img_data_params.path), 
                                expandvars_recursive(self.img_data_params.topic))
