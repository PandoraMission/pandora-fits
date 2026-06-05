# Standard library
import configparser  # noqa: E402
import logging  # noqa: E402
import os  # noqa
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version  # noqa
import pandoraaperture as pa  # noqa

# Third-party
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pandoraref as pr  # noqa: E402
from appdirs import user_config_dir, user_data_dir  # noqa: E402

# Third-party
from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402
from pandoraspacecraft import PandoraSpacecraft
import builtins

PACKAGEDIR = str(Path(__file__).resolve().parent)
PROJECTDIR = str(Path(__file__).resolve().parents[2])
FORMATSDIR = str(Path(PACKAGEDIR) / "formats") + os.sep
DOCSDIR = str(Path(PROJECTDIR) / "docs") + os.sep
TESTDIR = str(Path(PROJECTDIR) / "tests") + os.sep
PANDORASTYLE = [
    str(path) for path in (Path(PACKAGEDIR) / "data").glob("pandora.mplstyle")
]
logger = logging.getLogger("pandorafits")


def get_version():
    try:
        return version("pandorafits")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()


# Custom Logger with Rich
class PandoraLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO, logfile="DPC.log"):
        super().__init__(name, level)

        # --- Console (Rich) handler ---
        console = Console()
        console_handler = RichHandler(
            show_time=False, show_level=False, show_path=False, console=console
        )
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        # --- File handler ---
        file_handler = logging.FileHandler(logfile)
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

        # Attach both handlers
        self.addHandler(console_handler)
        self.addHandler(file_handler)


def _safe_makedirs(path: str | os.PathLike[str]) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_chmod(path: str | os.PathLike[str], mode: int) -> None:
    try:
        os.chmod(path, mode)
    except (AttributeError, NotImplementedError, OSError):
        pass


CONFIGDIR = user_config_dir("pandorafits")
_safe_makedirs(CONFIGDIR)
CONFIGPATH = os.path.join(CONFIGDIR, "config.ini")


def reset_config():
    """Set the config to defaults."""
    # use this function to set your default configuration parameters.
    config = configparser.ConfigParser()
    user_root = Path(user_data_dir("pandorafits"))
    config["SETTINGS"] = {
        "log_level": "INFO",
        "data_dir": str(user_root / "data"),
        "calendar_dir": str(user_root / "calendar"),
        "log_dir": str(user_root / "logs"),
        "level0_dir": str(user_root),
        "level1_dir": str(user_root / "level1"),
        "level2_dir": str(user_root / "level2"),
        "level3_dir": str(user_root / "level3"),
        "crsoftver": "v3.03",
    }
    with builtins.open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


def load_config() -> configparser.ConfigParser:
    """
    Loads the configuration file, creating it with defaults if it doesn't exist.

    Returns
    -------
    configparser.ConfigParser
        The loaded configuration.
    """

    config = configparser.ConfigParser()

    if not os.path.exists(CONFIGPATH):
        # Create default configuration
        reset_config()
    config.read(CONFIGPATH)
    return config


def save_config(config: configparser.ConfigParser) -> None:
    """
    Saves the configuration to the file.

    Parameters
    ----------
    config : configparser.ConfigParser
        The configuration to save.
    app_name : str
        Name of the application.
    """
    with builtins.open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


config = load_config()


def get_logger(name="pandorafits"):
    """Configure and return a logger with RichHandler."""

    jobid = os.environ.get("PBS_JOBID", "nojob")
    pid = os.getpid()
    logfile = os.path.join(LOG_DIR, f"pandoraDPC_{jobid}_{pid}.log")
    logger = PandoraLogger("pandora", logfile=logfile)
    return logger


LOG_DIR = config["SETTINGS"]["log_dir"]
_safe_makedirs(LOG_DIR)
_safe_chmod(LOG_DIR, 0o750)
logger = get_logger("pandorafits")
logger.setLevel(config["SETTINGS"]["log_level"])


# Use this to check that keys you expect are in the config file.
# If you update the config file and think users may be out of date
# add the config parameters to this loop to check and reset the config.
for key in [
    "level0_dir",
    "data_dir",
    "crsoftver",
    "level1_dir",
    "level2_dir",
    "level3_dir",
]:
    if key not in config["SETTINGS"]:
        logger.error(
            f"`{key}` missing from the `pandorafits` config file. Your configuration is being reset."
        )
        reset_config()
        config = load_config()

CALENDAR_DIR = config["SETTINGS"]["calendar_dir"]
LEVEL0_DIR = config["SETTINGS"]["level0_dir"]
DATA_DIR = config["SETTINGS"]["data_dir"]
CRSOFTVER = config["SETTINGS"]["crsoftver"]

LEVEL1_DIR = config["SETTINGS"]["level1_dir"]
LEVEL2_DIR = config["SETTINGS"]["level2_dir"]
LEVEL3_DIR = config["SETTINGS"]["level3_dir"]

[
    _safe_makedirs(dir)
    for dir in [LEVEL1_DIR, LEVEL2_DIR, LEVEL3_DIR]
]
[_safe_chmod(dir, 0o750) for dir in [LEVEL1_DIR, LEVEL2_DIR, LEVEL3_DIR]]


def display_config() -> pd.DataFrame:
    dfs = []
    for section in config.sections():
        df = pd.DataFrame(
            np.asarray(
                [(key, value) for key, value in dict(config[section]).items()]
            )
        )
        df["section"] = section
        df.columns = ["key", "value", "section"]
        df = df.set_index(["section", "key"])
        dfs.append(df)
    return pd.concat(dfs)


NIRDAReference = pr.NIRDAReference()
VISDAReference = pr.VISDAReference()

ps = PandoraSpacecraft()

from .io import *  # noqa
from .nirda import *  # noqa
from .visda import *  # noqa

# from .database import FileDataBase  # noqa
