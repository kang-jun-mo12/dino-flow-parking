from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StructureFlowConfig:
    # Window: 16 steps of 4 frames at 10 Hz.
    horizon: int = 16
    stride_frames: int = 4
    replan_steps: int = 16

    # Structure h: 0 F, 1 F->R, 2 R, 3 R->F, the encoding of k4_flow_solver.
    max_phases: int = 2
    coefficients: int = 12

    # Labels. Measurements give the vehicle centre; the bicycle model needs the rear axle.
    move_speed_mps: float = .1
    min_length_m: float = .02
    stop_length_m: float = .05
    rear_offset_m: float = 1.5
    samples: int = 64

    # Vehicle.
    curvature_scale: float = .5     # about tan(max steer)/wheelbase, 1/m
    max_speed_mps: float = 3.5

    # Fused BEV from the frozen DDP front end.
    bev_channels: int = 267
    token_grid: int = 16

    # Network.
    context_dim: int = 192
    temporal_channels: int = 48
    queries: int = 24
    heads: int = 6
    target_dim: int = 4
    ego_dim: int = 3
    coordinate_scale: float = 20.
    bev_attention: bool = True      # the phase U-Net reads scene tokens at its middle
    condition: str = 'bev'          # bev, seg, numeric or both
    scene_tokens: int = 512

    # Sampling.
    structures: int = 2
    candidates_per_structure: int = 4
    euler_steps: int = 8

    @property
    def window_frames(self):
        return self.horizon*self.stride_frames

    def to_dict(self):
        return asdict(self)


DEFAULT = StructureFlowConfig()
