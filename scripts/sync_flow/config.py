from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SyncFlowConfig:
    # Chunk. 16 is the project's choice; the released checkpoint does not pin it
    # because its U-Net is convolutional. Its cfg has data_interval=4, read here as
    # the waypoint spacing in 10 Hz frames: 0.4 s apart, 6.4 s horizon.
    horizon: int = 16
    frame_dt_s: float = .1
    stride_frames: int = 4
    channels: int = 4               # x, y, yaw (ego frame), signed speed

    # Fused-BEV contract, from last.ckpt: 264 fused channels + 3 segmentation
    # classes on the 16x16 grid its planner consumes (stop-gradient input).
    bev_channels: int = 267
    bev_size: int = 16
    token_grid: int = 16

    # Flow network.
    context_dim: int = 192
    width: int = 64
    heads: int = 6
    queries: int = 24
    target_dim: int = 4             # x/scale, y/scale, sin(yaw), cos(yaw)
    ego_dim: int = 3                # signed speed/max, steer, reverse

    # Vehicle and normalization.
    coordinate_scale: float = 20.
    max_speed_mps: float = 2.5
    wheelbase_m: float = 3.004647
    max_steer_rad: float = .9809930780518399

    # Decision loop.
    candidates: int = 8
    euler_steps: int = 8
    decision_dt_s: float = .2

    @property
    def waypoint_dt_s(self):
        return self.frame_dt_s*self.stride_frames

    @property
    def horizon_s(self):
        return self.horizon*self.waypoint_dt_s

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        if self.horizon % 4:
            raise ValueError('the U-Net pools the horizon twice by two')
        if self.stride_frames < 1 or self.decision_dt_s <= 0:
            raise ValueError('stride and decision period must be positive')


DEFAULT = SyncFlowConfig()
