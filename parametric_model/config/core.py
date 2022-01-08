from pathlib import Path
from pydantic import BaseModel
from strictyaml import YAML, load

import parametric_model


MODEL_ROOT = Path(parametric_model.__file__).parent
CONFIG_FILE_PATH = MODEL_ROOT / 'config.yml'


class AppConfig(BaseModel):
    app_name: str
    rc_test_data_file: str
    qp_region_test_data_file: str
    qp_mp_test_data_file: str
    lp_region_test_data_file: str
    lp_mp_test_data_file: str
    saved_model_file: str
    verif_test_size: int


class SolverConfig(BaseModel):
    lp_solver_path: str
    lp_solver_setting: str
    lp_activedual_tol: float
    qp_solver_path: str
    qp_solver_setting: str
    qp_activedual_tol: float


class RedundancyCheckerConfig(BaseModel):
    relax_tol: float
    zero_tol: float


class RegionGenConfig(BaseModel):
    lp_newregion_tol: float
    qp_newregion_tol: float
    max_iter_default: int


class OtherConfig(BaseModel):
    allclose_tol: float


class Config(BaseModel):
    app_config: AppConfig
    solver_config: SolverConfig
    redundancy_checker_config: RedundancyCheckerConfig
    regiongen_config: RegionGenConfig
    other_config: OtherConfig


def get_and_parse_config(config_file_path: Path=CONFIG_FILE_PATH):
    if config_file_path.is_file():
        with open(config_file_path, 'r') as config_file:
            parsed_config = load(config_file.read())
        return  parsed_config
    raise OSError('Did not find file at path: {config_file_path}')


def validate_config(parsed_config):
    _config = Config(
        app_config=AppConfig(**parsed_config.data),
        solver_config=SolverConfig(**parsed_config.data),
        redundancy_checker_config=RedundancyCheckerConfig(**parsed_config.data),
        regiongen_config = RegionGenConfig(**parsed_config.data),
        other_config=OtherConfig(**parsed_config.data))
    return _config


config = validate_config(get_and_parse_config())
