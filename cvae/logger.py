
import logging.config

def get_logger(name: str):

    logger_fn = logging.getLogger(name)
    log_write_file = 'cvae/logfile.txt'
    logging.basicConfig(filename=log_write_file, 
                        filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.DEBUG,
                        datefmt='%Y-%m-%d %H:%M:%S')

    return logger_fn


