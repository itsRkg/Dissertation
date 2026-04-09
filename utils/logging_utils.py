import logging
import os
from tqdm.notebook import tqdm


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logger(log_dir, experiment_name):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- file handler (normal) ----
    fh = logging.FileHandler(
        os.path.join(log_dir, f"{experiment_name}.log")
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # ---- tqdm-aware console handler ----
    th = TqdmLoggingHandler()
    th.setLevel(logging.INFO)
    th.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(th)

    return logger
