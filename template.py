import logging
import os
from pathlib import Path

# set logging messages settings
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')
#

project_name = "ml_project" # change the project name

list_of_files = [
    ".github/workflows/.gitkeep", # this is just to ensure the folder is uploaded
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "setup.py",
    "notebooks/.gitkeep",
    "templates/index.html",
]

for filepath in list_of_files:
    filepath = Path(filepath) # make sure path name is compatible with all OS
    filedir, filename = os.path.split(filepath)

    # create the folder first
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating folder {filedir} for the file {filename}.")

    # create the file in the folder
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"Skipping {filepath} as it already exists.")