# Standard library
import configparser  # noqa: E402
import logging  # noqa: E402
import os  # noqa
from importlib.metadata import PackageNotFoundError, version  # noqa

# Third-party
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pandoraref as pr  # noqa: E402
from appdirs import user_config_dir, user_data_dir  # noqa: E402

# Third-party
from rich.console import Console  # noqa: E402
from rich.logging import RichHandler  # noqa: E402

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
FORMATSDIR = f"{PACKAGEDIR}/formats/"
logger = logging.getLogger("pandorafits")


def get_version():
    try:
        return version("pandorafits")
    except PackageNotFoundError:
        return "unknown"


__version__ = get_version()


# Custom Logger with Rich
class PandoraLogger(logging.Logger):
    def __init__(self, name, level=logging.INFO):
        super().__init__(name, level)
        console = Console()
        self.handler = RichHandler(
            show_time=False, show_level=False, show_path=False, console=console
        )
        self.handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self.addHandler(self.handler)


def get_logger(name="pandorafits"):
    """Configure and return a logger with RichHandler."""
    return PandoraLogger(name)


CONFIGDIR = user_config_dir("pandorafits")
os.makedirs(CONFIGDIR, exist_ok=True)
CONFIGPATH = os.path.join(CONFIGDIR, "config.ini")

logger = get_logger("pandorafits")


CONFIGDIR = user_config_dir("pandorafits")
os.makedirs(CONFIGDIR, exist_ok=True)
CONFIGPATH = os.path.join(CONFIGDIR, "config.ini")


def reset_config():
    """Set the config to defaults."""
    # use this function to set your default configuration parameters.
    config = configparser.ConfigParser()
    config["SETTINGS"] = {
        "log_level": "INFO",
        "data_dir": "/Users/chedges/Downloads/",
        "level0_dir": user_data_dir("pandorafits"),
        "level1_dir": user_data_dir("pandorafits") + "/level1",
        "level2_dir": user_data_dir("pandorafits") + "/level2",
        "level3_dir": user_data_dir("pandorafits") + "/level3",
        "crsoftver": "v3.01",
    }
    with open(CONFIGPATH, "w") as configfile:
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
    with open(CONFIGPATH, "w") as configfile:
        config.write(configfile)


config = load_config()

# Use this to check that keys you expect are in the config file.
# If you update the config file and think users may be out of date
# add the config parameters to this loop to check and reset the config.
for key in ["level0_dir", "log_level"]:
    if key not in config["SETTINGS"]:
        logger.error(
            f"`{key}` missing from the `pandorafits` config file. Your configuration is being reset."
        )
        reset_config()
        config = load_config()

LEVEL0_DIR = config["SETTINGS"]["level0_dir"]
DATA_DIR = config["SETTINGS"]["data_dir"]
logger.setLevel(config["SETTINGS"]["log_level"])
CRSOFTVER = config["SETTINGS"]["crsoftver"]

LEVEL1_DIR = config["SETTINGS"]["level1_dir"]
LEVEL2_DIR = config["SETTINGS"]["level2_dir"]
LEVEL3_DIR = config["SETTINGS"]["level3_dir"]

[os.makedirs(dir, exist_ok=True) for dir in [LEVEL1_DIR, LEVEL2_DIR, LEVEL3_DIR]]
[os.chmod(dir, 0o750) for dir in [LEVEL1_DIR, LEVEL2_DIR, LEVEL3_DIR]]


def display_config() -> pd.DataFrame:
    dfs = []
    for section in config.sections():
        df = pd.DataFrame(
            np.asarray([(key, value) for key, value in dict(config[section]).items()])
        )
        df["section"] = section
        df.columns = ["key", "value", "section"]
        df = df.set_index(["section", "key"])
        dfs.append(df)
    return pd.concat(dfs)


NIRDAReference = pr.NIRDAReference()
VISDAReference = pr.VISDAReference()


from .nirda import *  # noqa
from .visda import *  # noqa
# from .database import FileDataBase  # noqa
