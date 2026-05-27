import argparse
import traceback
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

try:
    import wheeledlab_tasks
    print("[OK] wheeledlab_tasks imported")
except Exception:
    traceback.print_exc()
finally:
    simulation_app.close()