import logging
from datetime import datetime
from pathlib import Path

class LoggerUtility:
    @staticmethod
    def setup_logging(path_config: str):
        date = datetime.now()
        name = date.strftime('%Y%m%d') + '.log'
        logger = logging.getLogger('logger')
        if not logger.hasHandlers():
            logger.setLevel(logging.DEBUG)
            console_handler = logging.StreamHandler()
            cwd = Path.cwd()
            log_file = cwd.joinpath('logs')
            if not log_file.exists():
                log_file.mkdir(mode=0o770, parents=True, exist_ok=True)
                logger.info(f'Directory created in:{log_file}')
            else:
                logger.info(f'Directory already exists')
            file_handler = logging.FileHandler(log_file / name)
            console_handler.setLevel(logging.WARNING)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        return logger

    @staticmethod
    def get_logger():
        return logging.getLogger('logger')