# 프로그램 시작할 때
# 옵션 읽고
# Isaac 앱 켜고
# Navigation extension 켜고
# 환경 등록 파일 import 하는 역할.

"""
Boilerplate code for starting up IsaacLab backend
"""

import argparse
import sys


def startup(parser=None, prelaunch_callback=None, register_cfgs=True):
    from isaaclab.app import AppLauncher

    """
    Startup IsaacLab backend.

    Args:
        parser: argparse.ArgumentParser, optional
            Argument parser to add arguments to.
        prelaunch_callback: function, optional
            Function to be executed right before launching the app.
        register_cfgs: bool
            Whether to import/register wheeledlab task and run configs.

    Returns:
        simulation_app:
            Isaac Sim application instance.
        args_cli:
            Parsed command line arguments.
    """

    if parser is None:
        parser = argparse.ArgumentParser(description="Used Boilerplate Starter.")

    # IsaacLab / Isaac Sim launcher args 추가
    AppLauncher.add_app_launcher_args(parser)

    # train_rl.py의 -r 같은 인자는 parser가 먹고,
    # Hydra override 인자는 hydra_args로 따로 남김
    args_cli, hydra_args = parser.parse_known_args()

    if prelaunch_callback is not None:
        prelaunch_callback(args_cli)

    # rendering / video / camera 사용 가능하게 설정
    args_cli.enable_cameras = True

    # Hydra가 사용할 인자만 sys.argv에 남김
    sys.argv = [sys.argv[0]] + hydra_args

    # --------------------------------------------------
    # Launch Isaac Sim / Omniverse Kit
    # --------------------------------------------------
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # --------------------------------------------------
    # Wait until Kit / Extension Manager is stable
    # --------------------------------------------------
    for _ in range(120):
        simulation_app.update()

    # --------------------------------------------------
    # Enable Isaac Sim extensions automatically
    # --------------------------------------------------
    import omni.kit.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()

    EXTENSIONS = [
        # "isaacsim.ros2.bridge",       # ROS2 Bridge는 현재 자동 enable 보류
        "omni.anim.navigation.bundle",  # Navigation / NavMesh 관련
    ]

    for ext in EXTENSIONS:
        try:
            ext_manager.set_extension_enabled_immediate(ext, True)
            print(f"[INFO] Requested extension enable: {ext}")
        except Exception as e:
            print(f"[WARN] Failed to request extension {ext}: {e}")

    # extension 실제 로딩 반영 대기
    for _ in range(120):
        simulation_app.update()

    # extension 상태 확인
    for ext in EXTENSIONS:
        try:
            print(f"[CHECK] {ext} enabled = {ext_manager.is_extension_enabled(ext)}")
        except Exception as e:
            print(f"[CHECK] {ext} status check failed: {e}")

    # --------------------------------------------------
    # Register custom task / run configs
    # --------------------------------------------------
    if register_cfgs:
        import wheeledlab_tasks  # env configs
        import wheeledlab_rl.configs.runs  # run configs

    return simulation_app, args_cli