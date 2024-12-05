# tbbrdet_api
[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/thermal-bridges-rooftops-detector/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/thermal-bridges-rooftops-detector/job/main/)

DEEPaaS API for [TBBRDet model](https://github.com/emvollmer/TBBRDet).

To launch it, first install the package via the provided bash scripts, then run [deepaas](https://github.com/ai4os/DEEPaaS):
```bash
wget https://raw.githubusercontent.com/ai4os-hub/thermal-bridges-rooftops-detector/main/deployment_setup.sh
source deployment_setup.sh 	# this sets up the deployment (CUDA, CUDNN, Python3.6)
source install_TBBRDet.sh 	# this sets up the venv with all required packages and installs the both API and submodule TBBRDet as editable
deep-start
# Alternatively
deepaas-run --listen-ip 0.0.0.0
```
When re-deploying after initial setup, remember to activate the virtual environment before running deepaas:
```bash
source venv/bin/activate
deep-start
```


## Project structure
```
├── LICENSE                <- License file
│
├── README.md              <- The top-level README for developers using this project.
│
├── requirements.txt       <- The requirements file for reproducing the analysis environment, e.g.
│                             generated with `pip freeze > requirements.txt`
│
├── setup.py, setup.cfg    <- makes project pip installable (pip install -e .) so
│                             tbbrdet_api can be imported
│
├── data           <- Folder to download data to
│
├── models         <- Folder to save trained or downloaded models to
│
├── tbbrdet_api    <- Source code for the API to integrate the submodule TBBRDet with the platform.
│   │
│   ├── __init__.py        <- Makes tbbrdet_api a Python module
│   │
│   └── api.py             <- Main script for the integration with DEEP API
│   │
│   └── fields.py          <- Schema for frontend via Swagger UI
│   │
│   └── misc.py            <- Script containing helper functions
│
└── Jenkinsfile            <- Describes basic Jenkins CI/CD pipeline
```

## Testing

The testing process is automated via the tox library. To check style, coverage and security, run:

```bash
$ pip install tox
$ tox
```
