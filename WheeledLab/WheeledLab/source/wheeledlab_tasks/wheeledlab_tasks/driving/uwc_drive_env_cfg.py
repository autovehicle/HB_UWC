"""UWC path-following driving env.

목표:
- UWC 로봇이 미리 정의된 path 를 잘 추종하도록 학습
- 주행 가능 영역(drivable area) 밖으로는 절대 나가지 않음 -> 강한 종료 페널티
- 일정 시간 이상 멈춰 있어도 안 됨 -> stuck 종료 페널티
- 관측에는 BlindObs + 로봇 로컬 프레임의 lookahead waypoint 들을 추가

NOTE:
- nav_init은 현재 config 생성 시점이 아니라 env/stage 생성 이후 실행되어야 함.
- 따라서 이 파일에서는 navigation extension 로딩 여부만 전제로 하고,
  nav_init 직접 실행은 일단 비활성화한다.
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
    increase_reward_weight_over_time,
)

from wheeledlab_tasks.common import BlindObsCfg

from wheeledlab_assets import ROBOT_MODEL_CFG as UWC_CFG
from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR


##############################
###### COMMON CONSTANTS ######
##############################

MAX_SPEED = 2.0
MIN_SPEED = 0.3
STATIONARY_TIME_S = 2.0

NUM_LOOKAHEAD = 5
LOOKAHEAD_STEP = 0.5

CUSTOM_MAP_USD = f"{WHEELEDLAB_ASSETS_DATA_DIR}/map/map3.usd"


###################################
###### PLACEHOLDER MDP TERMS ######
###################################

def lookahead_waypoints_local(
    env: ManagerBasedEnv,
    num_points: int = NUM_LOOKAHEAD,
    step: float = LOOKAHEAD_STEP,
) -> torch.Tensor:
    """로봇 로컬 frame 기준 다음 num_points 개 waypoint 의 (x, y) 좌표."""
    return torch.zeros(env.num_envs, num_points * 2, device=env.device)


def path_cross_track_dist(env: ManagerBasedEnv) -> torch.Tensor:
    """현재 위치에서 path 까지의 최단 수직 거리."""
    return torch.zeros(env.num_envs, device=env.device)


def path_heading_error_sq(env: ManagerBasedEnv) -> torch.Tensor:
    """path 접선 방향과 robot heading 사이 각도 오차의 제곱."""
    return torch.zeros(env.num_envs, device=env.device)


def path_forward_progress(env: ManagerBasedEnv) -> torch.Tensor:
    """path 접선 방향 성분의 로봇 속도."""
    return torch.zeros(env.num_envs, device=env.device)


def is_outside_drivable_area(env: ManagerBasedEnv) -> torch.Tensor:
    """drivable area 외부에 있으면 True."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def is_stuck(
    env: ManagerBasedEnv,
    min_speed: float = MIN_SPEED,
    max_time: float = STATIONARY_TIME_S,
) -> torch.Tensor:
    """일정 시간 이상 정지 상태면 True."""
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def reset_root_state_along_path(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    pos_noise: float = 0.0,
    yaw_noise: float = 0.0,
):
    """drivable area 안의 path 위 임의의 점에서 로봇을 reset.

    현재 placeholder 구현은 origin reset.
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    n = len(env_ids)
    pos = torch.zeros(n, 3, device=env.device)

    quat = torch.zeros(n, 4, device=env.device)
    quat[:, 0] = 1.0  # identity quaternion, wxyz

    asset.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=env_ids)


def stationary_penalty(
    env: ManagerBasedEnv,
    min_speed: float = MIN_SPEED,
) -> torch.Tensor:
    """속도가 min_speed 보다 낮으면 1, 아니면 0."""
    lin_vel = mdp.base_lin_vel(env)
    ground_speed = torch.norm(lin_vel[..., :2], dim=-1)
    return torch.where(ground_speed < min_speed, 1.0, 0.0)


def vel_dist(
    env: ManagerBasedEnv,
    speed_target: float = MAX_SPEED,
    offset: float = -MAX_SPEED ** 2,
) -> torch.Tensor:
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
    """UWC 로봇 + 커스텀 맵 scene."""

    terrain = UWCDriveTerrainImporterCfg()

    robot: ArticulationCfg = UWC_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    def __post_init__(self):
        super().__post_init__()

        # 맵 z값에 따라 조정 필요
        self.robot.init_state = self.robot.init_state.replace(
            pos=(0.0, 0.0, 0.5)
        )


########################
###### ACTIONS #########
########################

@configclass
class UWCActionCfg:
    """UWC 4-wheel velocity action.

    현재는 디버깅용 4D wheel velocity action.
    추후에는 2D skid-steer action으로 바꾸는 것이 좋음.
    """

    wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=[".*wheel_joint"],
        scale=3.0,
        use_default_offset=False,
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
            params={
                "num_points": NUM_LOOKAHEAD,
                "step": LOOKAHEAD_STEP,
            },
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
    """학습 안정성을 위한 최소 이벤트.

    friction randomization / actuator gain randomization은
    UWC body/joint 이름과 IsaacLab 5.1 API에 맞춰 추후 재작성 필요.
    """

    # 조작감 확인 단계에서는 외란도 꺼두는 편이 안정적임.
    # 필요하면 나중에 다시 활성화.
    """
    push_robots_hf = EventTerm(
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
    """

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
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
    """Path-following 에 맞춘 보상."""

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

    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )

    out_of_drivable_area = DoneTerm(
        func=is_outside_drivable_area,
    )

    stuck = DoneTerm(
        func=is_stuck,
        params={
            "min_speed": MIN_SPEED,
            "max_time": STATIONARY_TIME_S,
        },
    )


######################
###### RL ENV ########
######################

@configclass
class UWCDriveRLEnvCfg(ManagerBasedRLEnvCfg):
    """UWC driving RL env config."""

    seed: int = 42

    # Scene
    scene: UWCDriveSceneCfg = UWCDriveSceneCfg(
        num_envs=1,
        env_spacing=0.0,
    )

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

        # sim timing: 200Hz physics / 50Hz control
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = 20

        # path 추종은 한 에피소드를 좀 더 길게
        self.episode_length_s = 10

        # 주의:
        # nav_init은 여기서 실행하지 않음.
        # 이 시점은 아직 env/stage가 실제로 생성되기 전 config 등록 단계라
        # NavMesh가 None일 가능성이 큼.
        #
        # nav_init은 추후 train_rl.py에서 env = gym.make(...) 이후로 이동해야 함.


######################
###### PLAY ENV ######
######################

@configclass
class UWCDrivePlayEnvCfg(UWCDriveRLEnvCfg):
    """평가/시연용."""

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