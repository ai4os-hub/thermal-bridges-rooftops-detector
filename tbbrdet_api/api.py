# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing the interfacing
tasks. In this way you don't mix your true code with DEEPaaS code and everything is
more modular. That is, if you need to write the predict() function in api.py, you
would import your true predict function and call it from here (with some processing /
postprocessing in between if needed).
For example:

    import mycustomfile

    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at a canonical exemplar
module [2].

[1]: https://docs.deep-hybrid-datacloud.eu/
[2]: https://github.com/deephdc/demo_app
"""
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import pkg_resources

from tbbrdet_api import configs, fields, misc
from tbbrdet_api.scripts import infer
from tbbrdet_api.scripts.train import main
from tbbrdet_api.misc import _catch_error

logger = logging.getLogger('__name__')


BASE_DIR = Path(__file__).resolve().parents[1]


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
        'authors': configs.MODEL_METADATA.get("author"),
        'description': configs.MODEL_METADATA.get("summary"),
        'license': configs.MODEL_METADATA.get("license"),
        'version': configs.MODEL_METADATA.get("version"),
        'checkpoints_local': misc.ls_local(),
        'checkpoints_remote': misc.ls_remote(),
        }
    logger.debug("Package model metadata: %d", metadata)
    return metadata
    # distros = list(pkg_resources.find_distributions(str(BASE_DIR), only=True))
    # if len(distros) == 0:
    #     raise Exception("No package found.")
    # pkg = distros[0]  # if several select first
    #
    # meta_fields = {
    #     "name": None,
    #     "version": None,
    #     "summary": None,
    #     "home-page": None,
    #     "author": None,
    #     "author-email": None,
    #     "license": None,
    # }
    # meta = {}
    # for line in pkg.get_metadata_lines("PKG-INFO"):
    #     line_low = line.lower()  # to avoid inconsistency due to letter cases
    #     for k in meta_fields:
    #         if line_low.startswith(k + ":"):
    #             _, value = line.split(": ", 1)
    #             meta[k] = value

    # return meta


def get_train_args():
    """
    Return the arguments that are needed to perform a  training.

    Returns:
        Dictionary of webargs fields.
      """
    train_args = fields.TrainArgsSchema().fields
    logger.debug("Web arguments: %d", train_args)
    return train_args


def get_predict_args():
    """
    Return the arguments that are needed to perform a  prediciton.

    Args:
        None

    Returns:
        Dictionary of webargs fields.
    """
    predict_args = fields.PredictArgsSchema().fields
    logger.debug("Web arguments: %d", predict_args)
    return predict_args


def train(**args):
    """
    Performs training on the dataset.
    Args:
        **args: keyword arguments from get_train_args.
    Returns:
        path to the trained model
    """
    assert not (args.get('resume_training', False) and not args.get('weights')), \
        "weights argument should not be empty when resume_training is True"
    timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    ckpt_path = os.path.join(configs.MODEL_DIR, timestamp)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    args['name'] = ckpt_path
    args['data_config'] = os.path.join(configs.DATA_PATH, args['data_config'])
    main(args)
    return {f'model was saved in {args["name"]}'}


def predict(**args):
    """
    Performs inference  on an input image.
    Args:
        **args:   keyword arguments from get_predict_args.
    Returns:
        either a json file or png image with bounding box
    """
    misc.download_model_from_nextcloud(args['timestamp'])

    args['weights'] = os.path.join(configs.MODEL_DIR, args['timestamp'], 'best_model.pth')

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in [args['input']]:
            shutil.copy(f.filename, tmpdir + F'/{f.original_filename}')
        args['input'] = [os.path.join(tmpdir, t) for t in os.listdir(tmpdir)]
        outputs, buffer = infer.main(args)

        if args['accept'] == 'image/png':
            return buffer
        else:
            return outputs


if __name__ == '__main__':
    args = {'model': 'fasterrcnn_convnext_small',
            'data_config': 'test_data/submarin.yaml',
            'use_train_aug': False,
            'device': True,
            'epochs': 1,
            'workers': 4,
            'batch': 1,
            'lr': 0.001,
            'imgsz': 640,
            'no_mosaic': True,
            'cosine_annealing': False,
            'weights': '/home/se1131/fasterrcnn_pytorch_api/models/2023-05-10_121810/last_model.pth',
            'resume_training': True,
            'square_training': False,
            'seed': 0
            }
    train(**args)

# def warm():
#     pass
#
#
# def get_predict_args():
#     return {}
#
#
# @_catch_error
# def predict(**kwargs):
#     return None
#
#
# def get_train_args():
#     return {}
#
#
# def train(**kwargs):
#     return None
