import logging
import sys


class LoggerProvider(object):
    __instance = None

    @staticmethod
    def get_instance():
        return LoggerProvider.__instance

    LOG_LEVELS = {
        "50": "CRITICAL",
        "40": "ERROR",
        "30": "WARNING",
        "20": "INFO",
        "10": "DEBUG"
    }

    def __init__(self, log_level=20):
        LoggerProvider.__instance = self
        self._log_level = int(log_level)
        self._handler = self._extract_handler_with_formatter(ColorFormatter())
        self._handler.setFormatter(ColorFormatter())

    def _extract_handler_with_formatter(self, formatter):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        return handler

    def get_logger(self, prefix):
        logger = logging.getLogger(prefix)
        logger.addHandler(self._handler)
        logger.setLevel(self._log_level)
        return logger


def get_log_level(self):
    return self.LOG_LEVELS[str(self._log_level)]


class ColorFormatter(logging.Formatter):
    FORMAT = "%(asctime)s %(levelname)s %(name)s$RESET - %(message)s$RESET"  # (%(filename)s:%(lineno)d)
    BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[1;%dm"
    BOLD_SEQ = "\033[1m"

    COLORS = {
        'CRITICAL': RED,
        'ERROR': RED,
        'WARNING': YELLOW,
        'INFO': BLUE,
        'DEBUG': CYAN
    }

    def __init__(self, use_color=True, log_format=FORMAT):
        self._use_color = use_color
        msg = self.formatter_msg(log_format, self._use_color)
        logging.Formatter.__init__(self, msg)

    def formatter_msg(self, msg, use_color=True):
        if use_color:
            msg = msg.replace("$RESET", self.RESET_SEQ).replace("$BOLD", self.BOLD_SEQ)
        else:
            msg = msg.replace("$RESET", "").replace("$BOLD", "")
        return msg

    def format(self, record):
        levelname = record.levelname
        if self._use_color and levelname in self.COLORS:
            fore_color = 30 + self.COLORS[levelname]
            levelname_color = self.COLOR_SEQ % fore_color + levelname
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)
