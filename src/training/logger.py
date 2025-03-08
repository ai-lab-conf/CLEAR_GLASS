import logging

#sets up a loggin configuration that logs messages both to the console and a file


def setup_logging(log_file, level, include_host=False):
    """
    ### Parameters

    - log_file (str): 
        The file path where log messages will be written. If `None`, log messages will not be written to a file.
    - level (int):
        The logging level to set for the root logger and all existing loggers. This should be one of the standard logging levels: `logging.DEBUG`, `logging.INFO`, `logging.WARNING`, `logging.ERROR`, `logging.CRITICAL`.

    - include_host (bool, optional, default=False):
        A flag indicating whether to include the hostname in log messages. When `True`, the hostname of the machine will be included in the log output.
    """
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)