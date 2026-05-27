# 실험 레시피 모음:
# drift 레시피
# visual 레시피
# elevation 레시피

from isaaclab.utils import configclass

from wheeledlab_rl.configs import (
    EnvSetup, RslRlRunConfig, RLTrainConfig, AgentSetup, LogConfig
)

@configclass
class RSS_UWC_DRIVE_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1,
        task_name="Isaac-UWC-Drive-RL-v0",
    )
    train = RLTrainConfig(
        num_iterations=4000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo",
        log=LogConfig(
            logs_dir="C:/Users/autonav009/HB_UWC/WheeledLab/WheeledLab/source/wheeledlab_rl/logs",
            run_name="uwc_drive_debug",
            model_save_dirname="models",
            video_interval=10000,
            no_wandb=True,
        ),
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point",
    )
    
@configclass
class RSS_DRIFT_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1024,
        task_name="Isaac-MushrDriftRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo",
        log=LogConfig(
            video_interval=15000
        ),
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

@configclass
class RSS_VISUAL_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=512,
        task_name="Isaac-MushrVisualRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

@configclass
class RSS_ELEV_CONFIG(RslRlRunConfig):
    env_setup = EnvSetup(
        num_envs=1024,
        task_name="Isaac-MushrElevationRL-v0"
    )
    train = RLTrainConfig(
        num_iterations=5000,
        rl_algo_lib="rsl",
        rl_algo_class="ppo"
    )
    agent_setup = AgentSetup(
        entry_point="rsl_rl_cfg_entry_point"
    )

    
