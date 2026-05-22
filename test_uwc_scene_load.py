import sys

# HB_UWC 패키지 경로 등록
sys.path.append(r"C:\Users\autonav009\Desktop\hb_uwc\WheeledLab\WheeledLab\source\wheeledlab_assets")
sys.path.append(r"C:\Users\autonav009\Desktop\hb_uwc\WheeledLab\WheeledLab\source\wheeledlab_tasks")
sys.path.append(r"C:\Users\autonav009\Desktop\hb_uwc\WheeledLab\WheeledLab\source\wheeledlab_rl")
sys.path.append(r"C:\Users\autonav009\Desktop\hb_uwc\WheeledLab\WheeledLab\source\wheeledlab")

from wheeledlab_rl.startup import startup

# Isaac Sim 실행
simulation_app, args_cli = startup(register_cfgs=False)

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene

# 네 파일 위치 기준
from wheeledlab_tasks.driving.uwc_drive_env_cfg import UWCDriveSceneCfg


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.005)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([4.0, -4.0, 3.0], [0.0, 0.0, 0.0])

    scene_cfg = UWCDriveSceneCfg(num_envs=1, env_spacing=0.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] UWC scene loaded: map + robot")

    while simulation_app.is_running():
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()