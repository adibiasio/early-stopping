import logging


def make_logger(name: str, verbosity: int, path: str | None = None) -> logging.Logger:
    """
    Makes a logger with the following default verbosity level.

    Parameters:
    -----------
    name: str
        The name of the new logger.
    verbosity: int
        The verbosity level of the logger.
        Expected to be in range 0 to 5, with the following meanings:
            NOTSET (5): Default level, which means no specific level is set.
            DEBUG (4): Detailed information, often used for diagnosing issues.
            INFO (3): General information about the program’s execution.
            WARNING (2): Used for warnings and potential issues.
            ERROR (1): Indicates a significant problem that needs attention.
            CRITICAL (0): Highest severity, typically indicating a serious problem
                that might prevent the program from continuing.
    path: str | None, default = None
        The path to the folder containing the output log file.
        If set to None, logs will not be outputted to any file.

    Returns:
    --------
    Logger:
        The logger object.
    """
    level = (5 - verbosity) * 10

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -- %(message)s"
    )

    # output to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # if path specified, also output to file
    if path:
        import os

        path = os.path.join(path, f"{name}.log")
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
