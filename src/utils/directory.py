import os


def verify_directory(directory: str) -> bool:
    return os.path.exists(directory)
