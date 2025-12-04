import shutil
import os

path = os.getenv("CHROMA_DIR", "./chroma_db")
if os.path.exists(path):
    shutil.rmtree(path)
    print(f"Removed {path}")
else:
    print(f"No Chroma directory at {path}")
