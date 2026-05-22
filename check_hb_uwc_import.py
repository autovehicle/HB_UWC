from wheeledlab_rl.startup import startup

simulation_app, args_cli = startup(register_cfgs=False)

from wheeledlab_assets import WHEELEDLAB_ASSETS_DATA_DIR
from wheeledlab_assets import ROBOT_MODEL_CFG

print("[INFO] wheeledlab_assets import OK")
print("[INFO] asset data dir:", WHEELEDLAB_ASSETS_DATA_DIR)
print("[INFO] ROBOT_MODEL_CFG import OK")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()