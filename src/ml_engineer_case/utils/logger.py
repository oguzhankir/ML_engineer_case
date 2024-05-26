import logging


class AppLogger:
    """
    Class for creating and retrieving logger instances.

    Attributes:
    - name (str): Name of the logger.
    """

    def __init__(self, name: str):
        """
        Initialize the AppLogger with the provided name.

        Args:
        - name (str): Name of the logger.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_logger(self):
        """
        Get the logger instance.

        Returns:
        - Logger: Logger instance.
        """
        return self.logger
