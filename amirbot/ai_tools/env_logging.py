import dotenv
import logging
import logging.config
import os
from rich.logging import RichHandler


def init_env_logging(env_path):
    dotenv.load_dotenv(dotenv_path=env_path)

    # Define the configuration file path based on the environment
    config_path = os.getenv('LOGGING_CONF_PATH')

    # Use the configuration file appropriate to the environment
    logging.config.fileConfig(config_path)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.DEBUG)
    logging.getLogger("httpcore.connection").setLevel(logging.DEBUG)
    logging.getLogger("httpcore.http11").setLevel(logging.DEBUG)
    logging.getLogger("openai._base_client").setLevel(logging.DEBUG)