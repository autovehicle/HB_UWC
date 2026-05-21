"""Test script to verify robot_model asset loads correctly in Isaac Lab."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test robot_model asset in Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from wheeledlab_assets import ROBOT_MODEL_CFG


def design_scene():
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    robot_cfg = ROBOT_MODEL_CFG.copy()
    robot_cfg.prim_path = "/World/robot_model"
    robot = Articulation(cfg=robot_cfg)

    return {"robot": robot}


def run_simulator(sim: SimulationContext, entities: dict):
    robot: Articulation = entities["robot"]

    sim.reset()
    robot.reset()

    print("[INFO] robot_model asset loaded successfully.")
    print(f"[INFO] Joint names: {robot.joint_names}")
    print(f"[INFO] Body names:  {robot.body_names}")

    for i in range(100):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    print("[INFO] Simulation completed successfully.")


def main():
    sim_cfg = sim_utils.SimulationCfg(device="cpu", dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.0])

    entities = design_scene()
    run_simulator(sim, entities)
    simulation_app.close()


if __name__ == "__main__":
    main()
