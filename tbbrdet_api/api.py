# -*- coding: utf-8 -*-
"""
Functions to integrate the submodule TBBRDet model with the DEEPaaS API.
This file is minimal, only performing the interfacing tasks, so as not to
mix the "true" code with DEEPaaS code.
"""
import logging
import os
from torch import cuda
from pathlib import Path
# from PIL import Image

from tbbrdet_api import configs, fields
from tbbrdet_api.scripts.train import main
from tbbrdet_api.scripts.infer import infer
from tbbrdet_api.misc import (
    _catch_error, extract_zst,
    setup_folder_structure, copy_file,
    ls_folders,
)

logger = logging.getLogger('__name__')


@_catch_error
def get_metadata():
    """
    DO NOT REMOVE - All modules should have a get_metadata() function
    with appropriate keys.

    Returns a dictionary containing metadata information about the module.

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    metadata = {
        'api_name': configs.API_METADATA.get("name"),
        'model_name': configs.MODEL_METADATA.get("name"),
        'api_authors': configs.API_METADATA.get("author"),
        'model_authors': configs.MODEL_METADATA.get("author"),
        'description': configs.MODEL_METADATA.get("summary"),
        'home_page': configs.API_METADATA.get("home_page"),
        'license': configs.API_METADATA.get("license"),
        'version': configs.API_METADATA.get("version"),
        'datasets_LOCAL_zipped': ls_folders(configs.DATA_PATH, '*.tar.zst'),
        'datasets_LOCAL_unpacked': ls_folders(configs.DATA_PATH, '*.npy'),
        'datasets_REMOTE_zipped': ls_folders(configs.REMOTE_PATH, '*.tar.zst'),
        'datasets_REMOTE_unpacked': ls_folders(configs.REMOTE_PATH, '*.npy'),
        'model_folders_train_LOCAL': ls_folders(configs.MODEL_PATH),
        'model_folders_train_REMOTE': ls_folders(configs.REMOTE_MODEL_PATH),
        'model_folders_infer_LOCAL': ls_folders(configs.MODEL_PATH,
                                                "best*.pth"),
        'model_folders_infer_REMOTE': ls_folders(configs.REMOTE_MODEL_PATH,
                                                 "best*.pth"),
    }
    logger.debug("Package model metadata: %s", metadata)
    return metadata


def get_train_args():
    """
    Return the arguments that are needed to perform a  training.

    Returns:
        Dictionary of webargs fields.
      """
    train_args = fields.TrainArgsSchema().fields
    logger.debug("Web arguments: %s", train_args)
    return train_args


def get_predict_args():
    """
    Return the arguments that are needed to perform a prediction.

    Returns:
        Dictionary of webargs fields.
    """
    predict_args = fields.PredictArgsSchema().fields
    logger.debug("Web arguments: %s", predict_args)
    return predict_args


def train(**args):
    """
    Performs training on the dataset.

    Args:
        **args: keyword arguments from get_train_args.
    Returns:
        path to the trained model
    """
    print("Training with user provided arguments:\n", args)  # logger.info

    if not args['device'] or (args['device'] and not cuda.is_available()):
        logger.error("Training requires a GPU. "
                     "Please ensure a GPU is available before training.")
        raise ValueError("Training requires a GPU. "
                         "Please ensure a GPU is available before training.")

    if not Path(args['dataset_path']).is_dir():
        logger.error(f"Provided dataset_path '{args['dataset_path']}' "
                     f"does not exist as a folder containing files.")
        raise ValueError(f"Provided dataset_path '{args['dataset_path']}' "
                         f"does not exist as a folder containing files.")

    # redefine DATA_PATH to the user provided argument
    configs.DATA_PATH = Path(args.get('dataset_path', configs.DATA_PATH))
    
    # check if provided dataset_path contains .tar.zst files to extract
    tar_zst_paths = sorted(configs.DATA_PATH.rglob("*.tar.zst"))
    json_paths = sorted(configs.DATA_PATH.glob("*.json"))

    if tar_zst_paths and json_paths:
        logger.info(f"Provided dataset_path '{configs.DATA_PATH}' "
                    f"contains .tar.zst files to extract.")

        # handle zipped image numpy files through extraction
        extract_zst(data_dir=configs.DATA_PATH)

        # setup folder structure and move all files where they belong for training
        setup_folder_structure(data_dir=configs.DATA_PATH)

    elif (all(folder in os.listdir(configs.DATA_PATH)
              for folder in ["train", "test"])
          and list(configs.DATA_PATH.rglob("*.json"))
          and list(configs.DATA_PATH.rglob("*.npy"))):

        logger.info(f"Data folder '{configs.DATA_PATH}' already contains "
                    f"required data structure with .npy and .json files, "
                    f"so no additional extracting is necessary.")

    else:
        logger.error(f"Provided dataset_path '{args['dataset_path']}' "
                     f"does not contain any files to download.")
        raise FileNotFoundError(
            f"Provided dataset_path '{args['dataset_path']}' does not contain "
            f"any .tar.zst files to download and no extracted files exist.")

    # training config definitions
    args['cfg_options'] = {
        'data_root': str(configs.DATA_PATH),
        'runner.max_epochs': args['epochs'],
        'data.samples_per_gpu': args['batch'],
        'data.workers_per_gpu': args['workers']
    }

    model_dir = main(args)

    return {f'Model and logs were saved to {model_dir}'}


def predict(**args):
    """
    Performs inference on an input image.

    Args:
        **args:   keyword arguments from get_predict_args.
    Returns:
        either a json file or png image with bounding box
    """
    print("Predicting with user provided arguments:\nargs")  # logger.info

    # define model-related paths
    try:
        model_dir = Path(args['predict_model_dir'])
        args['config_file'] = str(sorted(model_dir.glob("*.py"))[-1])
        args['checkpoint_file'] = str(sorted(model_dir.glob("best*.pth"))[-1])
    except IndexError as e:
        logger.error(
            f"No checkpoint or config file found in "
            f"{args['predict_model_dir']}! Error: %s", e, exc_info=True)
        raise IndexError(e)

    # define output directory regardless of whether it's remote or local
    args['out_dir'] = Path(Path(args['predict_model_dir']), "predictions")
    args['out_dir'].mkdir(parents=True, exist_ok=True)

    predict_list = infer(args)    # list of paths to prediction .png(s)

    if args['accept'] == 'application/json':
        message = {
            'result': f"Inference result(s) saved to {', '.join(predict_list)}"
        }
    elif args['accept'] == 'image/png':
        message = open(predict_list[0], "rb")

    else:
        raise ValueError(f"Accept type '{args['accept']}' is not supported.")

    return message


if __name__ == '__main__':
#    ex_args = {
#        'dataset_path': '/storage/tbbrdet/datasets/',
#        'architecture': 'swin',
#        'train_from': '/storage/tbbrdet/models/swin/coco/2023-05-10_103541/',
#        # 'scratch',
#        'device': True,
#        'epochs': 1,
#        'workers': 2,
#        'batch': 1,
#        'lr': 0.0001,
#        'seed': 42,
#        'eval': "bbox"
#    }
#    train(**ex_args)

    ex_args = {
        'input':
            '/storage/tbbrdet/DJI_0004_R.npy',
        'predict_model_dir':
            '/srv/thermal-bridges-rooftops-detector/models/swin/coco/2023-12-07_130038/',
        'colour_channel': 'both',
        'threshold': 0.3,
        'device': True,
        'accept': 'image/png'
    }
    predict(**ex_args)
