import logging

from parametric_model.config.core import MODEL_ROOT, config

logging.getLogger(config.app_config.app_name).addHandler(logging.NullHandler())


with open(MODEL_ROOT / "VERSION") as version_file:
    __version__ = version_file.read().strip()
