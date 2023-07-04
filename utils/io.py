import os

def ensure_dir(dir: str):
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)
    return dir