"""
This file gathers some utility functions for all other scripts
(get_metadata, training and inference).
"""

from functools import wraps
import logging
import subprocess
from subprocess import TimeoutExpired
import time
from pathlib import Path
import sys
import shutil

from aiohttp.web import HTTPBadRequest

from tbbrdet_api import configs

logger = logging.getLogger('__name__')
logger.setLevel(configs.LOG_LEVEL)      # previously: logging.DEBUG


def _catch_error(f):
    """
    Decorate API functions to return an error as HTTPBadRequest,
    in case it fails.
    """

    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise HTTPBadRequest(reason=e)

    return wrap


def _fields_to_dict(fields_in):
    """
    Function to convert marshmallow fields to dict()
    """
    dict_out = {}
    for k, v in fields_in.items():
        param = {}
        param["default"] = v.missing
        param["type"] = type(v.missing)
        param["required"] = getattr(v, "required", False)

        v_help = v.metadata["description"]
        if "enum" in v.metadata.keys():
            v_help = f"{v_help}. Choices: {v.metadata['enum']}"
        param["help"] = v_help

        dict_out[k] = param

    return dict_out


def set_log(log_dir):
    logging.basicConfig(
        # level=logging.DEBUG,
        format='%(message)s',
        # dateformat='%a, %d %b %Y %H:%M:%S',
        filename=f"{log_dir}/train.log",
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(configs.LOG_LEVEL)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def extract_zst(data_dir: Path = Path(configs.DATA_PATH)):
    """
    Extracting the files from the tar.zst files

    Args:
        data_dir (Path): Path to folder containing .tar.zst files
    """
    log_disk_usage("Begin extracting .tar.zst files")

    # define the timeout according to the data location
    # (outside the docker container will take much longer)
    if str(configs.BASE_PATH) in str(data_dir):
        timeout = 600
    else:
        timeout = 6000

    for zst_path in Path(data_dir).glob("**/*.tar.zst"):
        tar_command = ["tar", "-I", "zstd", "-xf",  # -v flag to print names
                       str(zst_path), "-C", str(data_dir)]

        run_subprocess(
            tar_command,
            process_message=f"unpacking '{zst_path.name}'",
            timeout=timeout
        )

        # delete zst_path file in destination directory to save space
        if data_dir in zst_path.parents:
            logger.info(f"Removing .tar.zst file '{zst_path.name}' "
                        f"after extraction to save storage space.")
            zst_path.unlink()


def ls_folders(directory: Path = configs.MODEL_PATH,
               pattern: str = "*latest.pth") -> list:
    """
    Utility to return a list of folders in a given directory that contain
    a file of a specific pattern.

    - local_model_folders = ls_folders(directory=configs.MODEL_PATH,
                                       pattern="*latest.pth")
    - remote_model_folders = ls_folders(directory=configs.REMOTE_MODEL_PATH,
                                        pattern="*latest.pth")

    Args:
        directory (Path): Path of the directory to scan
        pattern (str): The pattern to use for scanning

    Returns:
        list: list of relevant .pth file paths
    """
    logger.debug(f"Scanning through '{directory}' with pattern '{pattern}'")
    pth_list = sorted(set([d.parent for d in Path(directory).rglob(pattern)]))

    if pattern == "*.npy":
        filtered_paths = []
        for pth in pth_list:
            # check if "train" or "test" appears in any parent directory
            valid_parent = next(
                (str(p.parent) for p in pth.parents 
                 if "train" in p.name or "test" in p.name),
                None
            )
            if valid_parent:  # only add path if the condition is fulfilled
                filtered_paths.append(valid_parent)

        return sorted(set(filtered_paths))  # remove duplicates

    else:
        return [str(p) for p in pth_list]


def setup_folder_structure(data_dir: Path = Path(configs.DATA_PATH)):
    """
    Create and populate the test / train folder structure if it does
    not already exist.
    |--- test/
    |     |--- annotations/
    |     |--- images/
    |--- train/
    |     |--- annotations/
    |     |--- images/
    
    Args:
        data_dir (str or Path): Directory where the structure will be created.
    """
    # Get current files / folders in data_dir
    exist_paths = sorted(data_dir.iterdir())

    # Create new folders
    folders = [
        Path(data_dir, "test", "annotations"),
        Path(data_dir, "test", "images"),
        Path(data_dir, "train", "annotations"),
        Path(data_dir, "train", "images"),
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
    print(f"Folder structure created at: {data_dir}")
    
    # Move files
    assoc = {"train": ["100", "101", "102", "103", "104"],
             "test": ["105"]}
    
    for pth in exist_paths:
        dataset = next(
            (k for k, ids in assoc.items() if any(s in pth.stem for s in ids)),
            None
        )

        if dataset:
            if pth.is_dir():  # Image folders
                shutil.move(str(pth), str(Path(data_dir, dataset, "images")))

            elif pth.suffix == ".json":  # Annotation files
                shutil.move(str(pth), str(Path(data_dir, dataset, "annotations")))
        else:
            raise ValueError(
                f"File '{pth}' does not match train or test dataset names '{assoc}'."
            )


def get_weights_folder(data: dict):
    """
    Utility to get folder containing pretrained weights (i.e. COCO weights)
    to use in transfer learning.

    Args:
        data (dict): Arguments from fields.py (user inputs in swagger ui)

    Returns:
        Path to the folder containing pretrained weights
    """
    return Path(configs.REMOTE_MODEL_PATH, data['architecture'],
                data['train_from'], "pretrained_weights")


def copy_file(frompath: Path, topath: Path):
    """
    Copy a file (also to / from remote directory)

    Args:
        frompath (Path): The path to the file to be copied
        topath (Path): The path to the destination folder directory

    Raises:
        OSError: If the source isn't a directory
        FileNotFoundError: If the source file doesn't exist
    """
    frompath: Path = Path(frompath)
    topath: Path = Path(topath)

    if Path(topath, frompath.name).exists():
        print(f"Skipping copy of '{frompath}' as the file already "
              f"exists in '{topath}'!")   # logger.info
    else:
        try:
            print(f"Copying '{frompath}' to '{topath}'...")  # logger.info
            topath = shutil.copy(frompath, topath)
        except OSError as e:
            print(f"Directory not copied because {frompath} "
                  f"directory not a directory. Error: %s" % e)
        except FileNotFoundError as e:
            print(f"Error in copying from {frompath} to {topath}. "
                  f"Error: %s" % e)


def run_subprocess(command: list, process_message: str,
                   timeout: int = 600):
    """
    Function to run a subprocess command.
    Tox security issue with subprocess is ignored here using # nosec.

    Args:
        command (list): Command to be run.
        process_message (str): Message to be printed to the console.
        timeout (int): Time limit by which process is limited

    Raises:
        TimeoutExpired: If timeout exceeded
        Exception: If any other error occurred
    """
    log_disk_usage(f"Begin: {process_message}")
    str_command = " ".join(command)

    print(f"=================================\n"
          f"Running {process_message} command:\n'{str_command}'\n"
          f"=================================")  # logger.info

    try:
        process = subprocess.Popen(      # nosec
                command,
                stdout=subprocess.PIPE,  # Capture stdout
                stderr=subprocess.PIPE,  # Capture stderr
                universal_newlines=True,  # Return strings rather than bytes
        )
        return_code = process.wait(timeout=timeout)

        if return_code == 0:
            log_disk_usage(f"Finished: {process_message}")

        else:
            _, err = process.communicate()
            logger.error(f"Error while running '{str_command}' for {process_message}.\n"
                         f" Terminated with return code {return_code}.")  # log.error
            process.terminate()
            raise MemoryError(err)

    except TimeoutExpired:
        process.terminate()
        logger.error(f"Timeout during {process_message} while running"
                     f"\n'{str_command}'\n{timeout} seconds were exceeded.")
        raise TimeoutError

    except Exception as e:
        process.terminate()
        raise MemoryError(e)

    return


def log_disk_usage(process_message: str):
    """Log used disk space to the terminal with a process_message describing
    what has occurred.
    """
    disk_usage = round(
        sum(f.stat().st_size for f in configs.BASE_PATH.rglob('*')
        if f.is_file()) / (1024 ** 3), 2
    )
    print(f"{process_message} --- Repository currently takes up "
          f"{disk_usage} GB.")  # logger.info


if __name__ == '__main__':
    print("Remote directory path:", configs.REMOTE_MODEL_PATH)
