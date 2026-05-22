# 프로그램 시작할 때
# 옵션 읽고
# Isaac 앱 켜고
# 환경 등록 파일 import 하는 역할.

"""
Boilerplate code for starting up IsaacLab backend
"""

import argparse
import sys

def startup(parser=None, prelaunch_callback=None, register_cfgs=True):
    from isaaclab.app import AppLauncher
    '''
    Startup IsaacLab backend. Imports wheeled_gym environments optionally.
    Args:
        parser: argparse.ArgumentParser, optional, default=None
            Argument parser to add arguments to.
        prelaunch(args): function to be executed right before launching the app, optional, default=None
    Returns:
        simulation_app: omni.isaac.dynamic_control.DynamicControl, omni.isaac.dynamic_control._dynamic_control.DynamicControl
            Simulation app instance.
        args_cli: argparse.Namespace
            Parsed command line arguments.
    '''

    if parser is None:
        parser = argparse.ArgumentParser(description="Used Boilerplate Starter.")

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()

    if prelaunch_callback is not None:
        prelaunch_callback(args_cli)

    args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # --------------------------------------------------
    # Enable Isaac Sim extensions automatically
    # --------------------------------------------------
    try:
        from isaacsim.core.utils.extensions import enable_extension
    except ImportError:
        from omni.isaac.core.utils.extensions import enable_extension

    EXTENSIONS = [
        "isaacsim.ros2.bridge",       # ROS 2 Bridge
        "omni.anim.navigation.bundle",  # Navigation / NavMesh 관련
    ]

    for ext in EXTENSIONS:
        try:
            enable_extension(ext)
            print(f"[INFO] Enabled extension: {ext}")
        except Exception as e:
            print(f"[WARN] Failed to enable extension {ext}: {e}")
    
    # extension 로딩 반영용
    simulation_app.update()

    if register_cfgs:
        import wheeledlab_tasks # env configs
        import wheeledlab_rl.configs.runs # run configs

    return simulation_app, args_cli