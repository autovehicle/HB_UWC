"""UWC path-following driving env.

목표:
- UWC 로봇이 미리 정의된 path 를 잘 추종하도록 학습
- 주행 가능 영역(drivable area) 밖으로는 절대 나가지 않음 -> 강한 종료 페널티
- 일정 시간 이상 멈춰 있어도 안 됨 -> stuck 종료 페널티
- 관측에는 BlindObs + 로봇 로컬 프레임의 lookahead waypoint 들을 추가

NOTE: 아래 TODO(team) 표시된 부분들은 다른 팀원이 작업 중인 모듈과 연결될 자리입니다.
  - UWC 로봇 에셋 (wheeledlab_assets.UWC_CFG 등)
  - 커스텀 맵 USD 경로
  - drivable area / path 관련 함수 (path 샘플링, lookahead, cross-track, heading, drivable check)
실제 코드가 준비되면 placeholder import / 함수만 교체하면 됩니다.
"""

import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedEnv
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise
from isaaclab.assets import ArticulationCfg, RigidObject, AssetBaseCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    CurriculumTermCfg as CurrTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)

from wheeledlab.envs.mdp import (
    RCCarRWDActionCfg,
    increase_reward_weight_over_time,
)
from wheeledlab_tasks.common import BlindObsCfg

# TODO(team): UWC 로봇 에셋이 wheeledlab_assets 에 등록되면 아래로 교체
#   from wheeledlab_assets import UWC_CFG
# 지금은 MUSHR 에셋을 placeholder 로 사용해 로딩만 되게 함.
# from wheeledlab_assets import MUSHR_SUS_2WD_CFG as UWC_CFG  # placeholder
# -> 교체완료 (20260522 17:20)
from wheeledlab_assets import ROBOT_MODEL_CFG as UWC_CFG
from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR

##############################
###### COMMON CONSTANTS ######
##############################

MAX_SPEED = 2.0              # (m/s) 목표 속도 — drift 와 달리 안정 추종 위해 낮춤
MIN_SPEED = 0.3              # (m/s) 이 이하면 정지로 판정
STATIONARY_TIME_S = 2.0      # (s) 이 시간 이상 정지하면 stuck 종료

NUM_LOOKAHEAD = 5            # 관측용 lookahead waypoint 개수
LOOKAHEAD_STEP = 0.5         # (m) lookahead waypoint 사이 간격

# 커스텀 맵 USD 경로 — TODO(team) 으로 실제 경로 교체
# CUSTOM_MAP_USD = "/PATH/TO/CUSTOM_MAP.usd"
# -> 교체완료 (20260522 17:20)
CUSTOM_MAP_USD = f"{WHEELEDLAB_ASSETS_DATA_DIR}/map/map2.usd"

###################################
###### PLACEHOLDER MDP TERMS ######
###################################
# TODO(team): 아래 함수들은 path/drivable area 모듈이 완성되면 import 로 교체.
# 각 함수의 입출력 시그니처는 유지해 주세요.


def lookahead_waypoints_local(env: ManagerBasedEnv,
                              num_points: int = NUM_LOOKAHEAD,
                              step: float = LOOKAHEAD_STEP) -> torch.Tensor:
    """로봇 로컬 frame 기준 다음 num_points 개 waypoint 의 (x, y) 좌표.

    Returns:
        (num_envs, num_points * 2) tensor.
    """
    return torch.zeros(env.num_envs, num_points * 2, device=env.device)


def path_cross_track_dist(env: ManagerBasedEnv) -> torch.Tensor:
    """현재 위치에서 path 까지의 최단 수직 거리 (m, 항상 ≥0)."""
    return torch.zeros(env.num_envs, device=env.device)


def path_heading_error_sq(env: ManagerBasedEnv) -> torch.Tensor:
    """path 접선 방향과 robot heading 사이 각도 오차의 제곱 (rad^2)."""
    return torch.zeros(env.num_envs, device=env.device)


def path_forward_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """path 접선 방향 성분의 로봇 속도 (m/s). 양의 값일수록 path 를 잘 따라감."""
    return torch.zeros(env.num_envs, device=env.device)


def is_outside_drivable_area(env: ManagerBasedEnv) -> torch.Tensor:
    """drivable area 외부에 있으면 True. (num_envs,) bool tensor."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def is_stuck(env: ManagerBasedEnv,
             min_speed: float = MIN_SPEED,
             max_time: float = STATIONARY_TIME_S) -> torch.Tensor:
    """일정 시간 이상 정지 상태면 True. (num_envs,) bool tensor.

    실제 구현은 env state buffer 에 stationary timer 를 누적해야 함.
    placeholder 는 항상 False.
    """
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def reset_root_state_along_path(env: ManagerBasedEnv,
                                env_ids: torch.Tensor,
                                asset_cfg: SceneEntityCfg,
                                pos_noise: float = 0.0,
                                yaw_noise: float = 0.0):
    """drivable area 안의 path 위 임의의 점에서 로봇을 reset.

    placeholder 구현: 일단 origin 에서 reset.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    n = len(env_ids)
    pos = torch.zeros(n, 3, device=env.device)
    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # identity (w, x, y, z)
    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)


def stationary_penalty(env: ManagerBasedEnv,
                       min_speed: float = MIN_SPEED) -> torch.Tensor:
    """속도가 min_speed 보다 낮으면 1, 아니면 0 — 매 스텝 정지 페널티용."""
    lin_vel = mdp.base_lin_vel(env)
    ground_speed = torch.norm(lin_vel[..., :2], dim=-1)
    return torch.where(ground_speed < min_speed, 1.0, 0.0)


def vel_dist(env: ManagerBasedEnv,
             speed_target: float = MAX_SPEED,
             offset: float = -MAX_SPEED ** 2) -> torch.Tensor:
    lin_vel = mdp.base_lin_vel(env)
    ground_speed = torch.norm(lin_vel[..., :2], dim=-1)
    return (ground_speed - speed_target) ** 2 + offset


###################
###### SCENE ######
###################

@configclass
class UWCDriveTerrainImporterCfg(TerrainImporterCfg):
    """커스텀 맵 USD 를 terrain 으로 import."""

    height = 0.0
    prim_path = "/World/ground"
    # TODO(team): 실제 커스텀 맵 USD 경로로 교체
    terrain_type = "usd"
    usd_path = CUSTOM_MAP_USD
    collision_group = -1
    physics_material = sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=0.9,
    )
    debug_vis = False


@configclass
class UWCDriveSceneCfg(InteractiveSceneCfg):
    """UWC 로봇 + 커스텀 맵 scene (센서는 추후 추가)."""

    terrain = UWCDriveTerrainImporterCfg()

    robot: ArticulationCfg = UWC_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

# 맵 z값에 따라 조정 필요
    def __post_init__(self):
        super().__post_init__()
        self.robot.init_state = self.robot.init_state.replace(pos=(0.0, 0.0, 0.5))


########################
###### ACTIONS #########
########################

@configclass
class UWCActionCfg:
    """TODO(team): UWC 로봇 실제 joint 이름과 차체 치수로 교체."""

    throttle_steer = RCCarRWDActionCfg(
        wheel_joint_names=[
            "back_left_wheel_throttle",     # placeholder
            "back_right_wheel_throttle",    # placeholder
        ],
        steering_joint_names=[
            "front_left_wheel_steer",       # placeholder
            "front_right_wheel_steer",      # placeholder
        ],
        base_length=0.325,    # placeholder (m)
        base_width=0.2,       # placeholder (m)
        wheel_radius=0.05,    # placeholder (m)
        scale=(MAX_SPEED, 0.488),
        no_reverse=True,
        bounding_strategy="clip",
        asset_name="robot",
    )


############################
###### OBSERVATIONS ########
############################

@configclass
class UWCDriveObsCfg(BlindObsCfg):
    """BlindObs + 로봇 로컬 frame lookahead waypoints."""

    @configclass
    class PolicyCfg(BlindObsCfg.PolicyCfg):

        lookahead_term = ObsTerm(
            func=lookahead_waypoints_local,
            params={"num_points": NUM_LOOKAHEAD, "step": LOOKAHEAD_STEP},
            noise=Gnoise(mean=0.0, std=0.05),
        )

    policy: PolicyCfg = PolicyCfg()


#####################
###### EVENTS #######
#####################

@configclass
class UWCDriveEventsCfg:

    reset_root_state = EventTerm(
        func=reset_root_state_along_path,
        params={
            "pos_noise": 0.3,
            "yaw_noise": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
        },
        mode="reset",
    )


@configclass
class UWCDriveEventsRandomCfg(UWCDriveEventsCfg):
    """학습 안정성을 위한 도메인 무작위화 (UWC joint/body 이름으로 교체 필요)."""

    change_wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.7, 1.0),   # path 추종이라 마찰 충분히 줌
            "dynamic_friction_range": (0.7, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 20,
            # TODO(team): UWC wheel body 이름 패턴으로 교체
            "asset_cfg": SceneEntityCfg("robot", body_names=".*wheel_link"),
            "make_consistent": True,
        },
    )

    randomize_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            # TODO(team): UWC throttle actuator joint 이름 패턴으로 교체
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*back.*throttle"]),
            "damping_distribution_params": (10.0, 50.0),
            "operation": "abs",
        },
    )

    push_robots_hf = EventTerm(  # 고주파 소규모 외란
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.1, 0.4),
        params={
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.03, 0.03),
                "yaw": (-0.3, 0.3),
            },
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            # TODO(team): UWC base body 이름으로 교체
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.3, 0.5),
            "operation": "add",
            "distribution": "uniform",
        },
    )


######################
###### REWARDS #######
######################

@configclass
class UWCDriveRewardsCfg:
    """Path-following 에 맞춘 보상.
    - forward_progress: path 접선 방향 속도 (+)
    - vel:              목표 속도와의 거리 (-)
    - cross_track:      path 와의 수직 거리 (-)
    - heading:          heading 오차^2 (-)
    - stationary_pen:   정지 상태 매 스텝 (-)
    - out_of_bounds:    drivable area 이탈 종료 시 큰 페널티 (-)
    - stuck_pen:        stuck 종료 시 페널티 (-)
    """

    forward_progress = RewTerm(
        func=path_forward_progress,
        weight=30.0,
    )

    vel = RewTerm(
        func=vel_dist,
        weight=-3.0,
        params={"speed_target": MAX_SPEED},
    )

    cross_track = RewTerm(
        func=path_cross_track_dist,
        weight=-50.0,
    )

    heading = RewTerm(
        func=path_heading_error_sq,
        weight=-10.0,
    )

    stationary_pen = RewTerm(
        func=stationary_penalty,
        weight=-20.0,
        params={"min_speed": MIN_SPEED},
    )

    out_of_bounds_pen = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": ["out_of_drivable_area"]},
        weight=-5000.0,
    )

    stuck_pen = RewTerm(
        func=mdp.rewards.is_terminated_term,
        params={"term_keys": ["stuck"]},
        weight=-500.0,
    )


########################
###### CURRICULUM ######
########################

@configclass
class UWCDriveCurriculumCfg:

    more_term_pens = CurrTerm(
        func=increase_reward_weight_over_time,
        params={
            "reward_term_name": "out_of_bounds_pen",
            "increase": -1000.0,
            "episodes_per_increase": 50,
            "max_increases": 5,
        },
    )


##########################
###### TERMINATION #######
##########################

@configclass
class UWCDriveTerminationsCfg:

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    out_of_drivable_area = DoneTerm(
        func=is_outside_drivable_area,
    )

    stuck = DoneTerm(
        func=is_stuck,
        params={"min_speed": MIN_SPEED, "max_time": STATIONARY_TIME_S},
    )


######################
###### RL ENV ########
######################

@configclass
class UWCDriveRLEnvCfg(ManagerBasedRLEnvCfg):

    seed: int = 42
    num_envs: int = 1024
    env_spacing: float = 0.0

    # Basic Settings
    observations: UWCDriveObsCfg = UWCDriveObsCfg()
    actions: UWCActionCfg = UWCActionCfg()

    # MDP Settings
    rewards: UWCDriveRewardsCfg = UWCDriveRewardsCfg()
    events: UWCDriveEventsCfg = UWCDriveEventsRandomCfg()
    terminations: UWCDriveTerminationsCfg = UWCDriveTerminationsCfg()
    curriculum: UWCDriveCurriculumCfg = UWCDriveCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # viewer
        self.viewer.eye = [4.0, -4.0, 4.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]

        # sim timing — drift 와 동일하게 200Hz physics / 50Hz control
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = 20
        # path 추종은 한 에피소드를 좀 더 길게
        self.episode_length_s = 10

        self.actions.throttle_steer.scale = (MAX_SPEED, 0.488)

        self.observations.policy.enable_corruption = True

        # Scene
        self.scene = UWCDriveSceneCfg(
            num_envs=self.num_envs, env_spacing=self.env_spacing,
        )


######################
###### PLAY ENV ######
######################

@configclass
class UWCDrivePlayEnvCfg(UWCDriveRLEnvCfg):
    """평가/시연용 — 종료/보상/커리큘럼 끔, reset 노이즈 없음."""

    events: UWCDriveEventsCfg = UWCDriveEventsRandomCfg(
        reset_root_state=EventTerm(
            func=reset_root_state_along_path,
            params={
                "pos_noise": 0.0,
                "yaw_noise": 0.0,
                "asset_cfg": SceneEntityCfg("robot"),
            },
            mode="reset",
        )
    )

    rewards: UWCDriveRewardsCfg = None
    terminations: UWCDriveTerminationsCfg = None
    curriculum: UWCDriveCurriculumCfg = None

    def __post_init__(self):
        super().__post_init__()
