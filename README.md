# PSDL-Anomaly-Detection

Official repository for the implementation of Patchwise Sparse Dictionary Learning from Pre-Trained Neural Network Activation Maps paper.

### Environment:

Docker folder contains Dockerfile to build the environment starting from official pytorch image.
Docker container settings: --entrypoint -v <path-to-the-project-folder>:/opt/project -v <path-to-the-parent-directory-of-dataset>:/data --rm --gpus all

### Configuration parameters:

Config.py file contains the following parameters:

- DATA_PATH = path to the folder containing the selected dataset
- SAVE_PATH = path to folder to save dictionary and error distributions
- ARCH = type of pre-trained model (available architectures: resnet18, wide_resnet50_2)
- TYPE = type of PSDL method: based on a single or multiple activation maps
- LAYER  = (in case of TYPE=SINGLE) layer of the activation maps used as signals for dictionary learning
- ONLY_TRAINING = whether to only train or perform also evaluation
- IMGS_NUMBER = number of images to be printed during evaluation
- SPARSE = sparsity level of OMP algorithm
- K_SVD_ITERATIONS = number total of iterations of K-SVD algorithm
- SEED = random seed


### Remark on Datasets
This code is not provided with built-in downloader of the used dataset. It is on the user to collect and provide valid paths in config.py.
In order to use MTD Datasets, folders and sub-folders should be arranged in a MVTec-like fashion: 
- dataset_folder
  - class1
  - class2
    - ground_truth
      - defects1
        - img1.xxx
        - img2.xxx
        - ...
    - test
      - defects1
        - img1.xxx
        - img2.xxx
        - ...
      - defects2
        - ...
      - good
        - img3.xxx
        - img4.xxx
        - ...
    - train
      - good
        - img5.xxx
        - img6.xxx
        - ...
  - ...

The code is inspired to [this](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master) unofficial implementation of PaDiM.

Algorithms for sparse representation and learning dictionary come from [sparse-land-tools](https://github.com/fubel/sparselandtools).

PaDiM and PatchCore are an authors reviewed versions of unoffical implementations [1](https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master), [2](https://github.com/hcw-00/PatchCore_anomaly_detection).