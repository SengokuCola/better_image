from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def plugin_module():
    spec = spec_from_file_location("better_image_release", Path(__file__).parents[1] / "plugin.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
