import argparse
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

print("[DEBUG 00] launching Isaac Sim...")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[DEBUG 01] Isaac Sim launched")

try:
    print("[DEBUG 02] importing gymnasium...")
    import gymnasium as gym
    print("[DEBUG 03] gymnasium imported")

    print("[DEBUG 04] importing wheeledlab_tasks...")
    import wheeledlab_tasks
    print("[DEBUG 05] wheeledlab_tasks imported")

    print("[DEBUG 06] importing parse_env_cfg...")
    from isaaclab_tasks.utils import parse_env_cfg
    print("[DEBUG 07] parse_env_cfg imported")

    task_name = "Isaac-UWC-Drive-RL-v0"

    print("[DEBUG 08] parsing env cfg...")
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    print("[DEBUG 09] env cfg parsed")

    print("[DEBUG 10] making env with cfg...")
    env = gym.make(task_name, cfg=env_cfg)
    print("[DEBUG 11] env created successfully")

    print("[DEBUG 12] reset env")
    env.reset()
    print("[DEBUG 13] reset success")

    for i in range(10):
        print(f"[DEBUG 14] step {i}")
        action = env.action_space.sample()
        env.step(action)
        simulation_app.update()

    print("[DEBUG 15] closing env")
    env.close()
    print("[DEBUG 16] env closed")

except Exception:
    print("[DEBUG ERROR] Python exception occurred:")
    traceback.print_exc()

finally:
    print("[DEBUG 99] closing simulation app")
    simulation_app.close()