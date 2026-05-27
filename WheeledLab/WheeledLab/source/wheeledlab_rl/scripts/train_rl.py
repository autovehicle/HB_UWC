###################################
###### BEGIN ISAACLAB SPINUP ######
###################################

# 전체 진행 총괄:
# 어떤 레시피 쓸지 받음 (-r)
# 환경 생성
# 학습기(PPO) 실행
# 로그/영상/저장 처리


from wheeledlab_rl.startup import startup
import argparse
parser = argparse.ArgumentParser(description="Train an RL Agent in WheeledLab.")
parser.add_argument('-r', "--run-config-name", type=str, default="RSS_DRIFT_CONFIG", help="Run in headless mode.")
simulation_app, args_cli = startup(parser=parser)

#######################
###### END SETUP ######
#######################

import gymnasium as gym
import os

from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

try:
    from isaaclab.utils.io import dump_pickle
except ImportError:
    import os
    import pickle

    def dump_pickle(filename, data):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(data, f)

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from wheeledlab_rl.configs import RunConfig
from wheeledlab_rl import WHEELEDLAB_RL_LOGS_DIR

from wheeledlab_rl.utils import (
    OnPolicyRunner as ModifiedRslRunner,
    CustomRecordVideo,
    hydra_run_config,
    ClipAction,
)

def run_nav_init_after_env_created():
    """Run UWC nav/path generation after the stage, map, and robot are created."""
    try:
        # stage/app이 실제로 업데이트되도록 몇 프레임 대기
        for _ in range(30):
            simulation_app.update()

        from wheeledlab_tasks.driving.nav_init import run_nav_init, NavInitCfg

        nav_cfg = NavInitCfg()
        run_nav_init(nav_cfg)

        # nav_init이 debug prim/path를 stage에 그리거나 저장하는 경우 반영 대기
        for _ in range(30):
            simulation_app.update()

        print("[INFO] nav_init completed after env creation.")

    except Exception as e:
        print(f"[WARN] nav_init failed after env creation: {e}")

@hydra_run_config(run_config_name=args_cli.run_config_name)
def main(run_cfg: RunConfig): # TODO: Add SB3 config support

    #################
    #### LOGGING ####
    #################

    ##### Aliasing Configs #####
    env_cfg = run_cfg.env
    agent_cfg = run_cfg.agent
    train_cfg = run_cfg.train
    log_cfg = train_cfg.log
    env_setup = run_cfg.env_setup

    if not log_cfg.no_wandb:
        import wandb
        run = wandb.init(
            project=log_cfg.wandb_project,
        )
        log_cfg.run_name = wandb.run.name

    if not os.path.exists(log_cfg.model_save_path):
        os.makedirs(log_cfg.model_save_path)

    ## UPDATE CONFIGS WANDB ##
    if not log_cfg.no_wandb:
        wandb.config.update(run_cfg.to_dict())

    # Save the config file
    if not log_cfg.no_log:
        dump_yaml(os.path.join(log_cfg.run_log_dir, "run_config.yaml"), run_cfg)
        dump_pickle(os.path.join(log_cfg.run_log_dir, "run_config.pkl"), run_cfg)

    ############################
    #### CREATE ENVIRONMENT ####
    ############################

    env = gym.make(env_setup.task_name, cfg=env_cfg, render_mode="rgb_array" if log_cfg.video else None)

    run_nav_init_after_env_created()

    ####### INSTANTIATE ENV #######
    env.action_space.low = -1.
    env.action_space.high = 1.
    env = ClipAction(env)

    # Wrap the environment in recorder
    if log_cfg.video:
        video_kwargs = {
            "video_folder": os.path.join(log_cfg.run_log_dir, "videos"),
            "step_trigger": lambda step: step % log_cfg.video_interval == 0,
            "video_length": log_cfg.video_length,
            "disable_logger": True,
            "enable_wandb": not log_cfg.no_wandb,
            "video_resolution": log_cfg.video_resolution,
            "video_crf": log_cfg.video_crf,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = CustomRecordVideo(env, **video_kwargs)

    # TODO: add back support for SB3
    env = RslRlVecEnvWrapper(env)

    agent_cfg_dict = agent_cfg.to_dict()

# rsl-rl 5.x MLPModel does not accept this field.
    legacy_model_keys = [
        "stochastic",
        "init_noise_std",
        "noise_std_type",
        "state_dependent_std",
    ]

    for key in ("actor", "critic"):
        if key in agent_cfg_dict:
            for legacy_key in legacy_model_keys:
                agent_cfg_dict[key].pop(legacy_key, None)

    runner = ModifiedRslRunner(env, agent_cfg_dict, log_cfg, device=train_cfg.device)
    ##### LOAD EXISTING RUN? #####

    if train_cfg.load_run is not None:
        chkpt = "model_"
        if train_cfg.load_run_checkpoint > 0:
            chkpt = f"{chkpt}{train_cfg.load_run_checkpoint}"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(WHEELEDLAB_RL_LOGS_DIR, run_dir=train_cfg.load_run,
                                        other_dirs=["models"], checkpoint=f"{chkpt}.*")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    #################
    ##### TRAIN #####
    #################

    env.seed(agent_cfg.seed)
    env.common_step_counter = train_cfg.set_env_step # For continuing curriculums
    runner.learn(num_learning_iterations=train_cfg.num_iterations)

    if not log_cfg.no_wandb:
        run.finish()
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()