import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

from . import WHEELEDLAB_ASSETS_DATA_DIR

##
# Actuator Configuration
##

# robot_model은 steering joint 없이 4개의 wheel joint로 구동하는
# 4WD 스키드스티어 방식 로봇입니다.
ROBOT_MODEL_ACTUATOR_CFG = {
    "wheel_joints": DCMotorCfg(
        joint_names_expr=[".*wheel_joint"],  # 4개 wheel joint 전체 매칭
         # 모터 토크 관련
        saturation_effort=12.0,
        effort_limit=6.0,

        # 휠 최대 각속도
        # 기존 400은 너무 큼. 50 rad/s 정도면 저속~중속 테스트용으로 적당.
        velocity_limit=50.0,

        # 휠은 위치제어가 아니라 속도/토크 기반 주행이므로 stiffness는 0 유지
        stiffness=0.0,

        # 기존 100은 너무 큼. 바퀴 회전 저항/속도 추종 안정용으로 낮춤
        damping=8.0,

        # 조인트 자체 마찰은 너무 크게 주면 바퀴가 둔해짐
        friction=0.05,
    ),
}

##
# Initial State Configuration
##

_ZERO_INIT_STATES = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "left_front_wheel_joint": 0.0,
        "left_rear_wheel_joint": 0.0,
        "right_front_wheel_joint": 0.0,
        "right_rear_wheel_joint": 0.0,
    },
)

##
# Robot Configuration
##

ROBOT_MODEL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{WHEELEDLAB_ASSETS_DATA_DIR}/Robots/UWC/robot_model.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=50.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=5.0,
            max_contact_impulse=None,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=6,
            solver_velocity_iteration_count=2,
            sleep_threshold=0.001,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=_ZERO_INIT_STATES,
    actuators=ROBOT_MODEL_ACTUATOR_CFG,
)
