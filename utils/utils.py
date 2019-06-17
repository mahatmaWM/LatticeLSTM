import logging


def configure_logging(level=logging.INFO):
    format = '%(asctime)s %(filename)s:%(lineno)d %(levelname)s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(level=level, format=format, datefmt=datefmt)
