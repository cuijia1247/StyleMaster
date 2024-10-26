import logging
import time

logger = logging.getLogger("my_logger")
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
log_name = current_time +'.log'
filehandler = logging.FileHandler("./log/" + log_name)
filehandler.setFormatter(formatter)
logger.addHandler(filehandler)

logger.info("Hello, logging world.")