import logging
import os
import time

def initLogging(logFilename, logLevel=logging.INFO):
    """Init for logging."""
    logger = logging.getLogger("")

    if not logger.handlers:
        logging.basicConfig(
            level=logLevel,
            format="[%(asctime)s-%(levelname)s] %(message)s",
            datefmt="%y-%m-%d %H:%M:%S",
            filename=logFilename,
            filemode="w",
        )
        console = logging.StreamHandler()
        console.setLevel(logLevel)
        formatter = logging.Formatter("[%(asctime)s-%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        logger.addHandler(console)

def init_ckpt_path(loglevel=logging.INFO):
    taskfolder = f"./results/"
    if not os.path.exists(taskfolder):
        os.makedirs(taskfolder)
    datafmt = time.strftime("%Y%m%d_%H%M%S")

    log_dir = f"./logs/{datafmt}.log"
    initLogging(log_dir, loglevel)
    ckpt_path = f"{taskfolder}/{datafmt}.pt"
    return ckpt_path