# 에셋(로봇/지형) 폴더 경로와 버전 정보를 한 곳에 모아둠.

import os
import toml

# Conveniences to other module directories via relative paths
WHEELEDLAB_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
"""Path to the extension source directory."""

WHEELEDLAB_ASSETS_DATA_DIR = os.path.join(WHEELEDLAB_ASSETS_EXT_DIR, "data")
"""Path to the extension data directory."""

WHEELEDLAB_ASSETS_METADATA = toml.load(os.path.join(WHEELEDLAB_ASSETS_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from the extension.toml file."""

# Configure the module-level variables
__version__ = WHEELEDLAB_ASSETS_METADATA["package"]["version"]

# from .mushr import *
# from .f1tenth import *
from .robot_model import *