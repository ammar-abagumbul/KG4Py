import logging

def setup_logging(level="INFO", log_format="%(asctime)s - %(levelname)s - %(message)s"):
    """
    Set up centralized logging configuration.

    Args:
        level (str): Logging level (e.g., "DEBUG", "INFO").
        log_format (str): Format for log messages.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # Logs to console
            logging.FileHandler("application.log")  # Logs to a file
        ]
    )

# Example usage
if __name__ == "__main__":
    setup_logging()
    logging.info("Logging setup complete.")