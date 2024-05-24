import logging
import sys

from colorama import Fore, Style


class ColorfulFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno == logging.DEBUG:
            prefix = Style.BRIGHT + Fore.CYAN
        elif record.levelno == logging.INFO:
            prefix = Style.BRIGHT + Fore.GREEN
        elif record.levelno == logging.WARNING:
            prefix = Style.BRIGHT + Fore.YELLOW
        elif record.levelno == logging.ERROR:
            prefix = Style.BRIGHT + Fore.RED
        else:
            prefix = Style.RESET_ALL
        msg = logging.Formatter.format(self, record)
        return prefix + msg + Style.RESET_ALL


def get_logger(name):
    logger = logging.getLogger(name)

    # Set the level of logger to INFO
    logger.setLevel(logging.INFO)

    # Create a stream handler that logs to stdout
    handler = logging.StreamHandler(sys.stdout)

    # Set the level of the stream handler to INFO
    handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = ColorfulFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the formatter for the handler
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger
