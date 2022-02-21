import ksdd
import mvtec
import mtd
import btad
import random_mvtec

DATA_PATH = "/data"
SAVE_PATH = "mvtec/mvtec_result"
# Dataset and class names to be used
DATASET = mvtec.MVTecDataset
CLASS_NAMES = mvtec.CLASS_NAMES
# Available architectures: resnet18, wide_resnet50_2, efficientnet-b5, efficientnet-b6, efficientnet-b7
ARCH = "efficientnet-b6"
# Available type mode: SINGLE, MULTIPLE. It will run PSDL with either one or multiple layers
TYPE = "SINGLE"
# In case of SINGLE, specify the LAYER (layer1, layer2, layer3);
# check utils.build_model() for more information about available layers
LAYER = 'layer3'
# Whether to execute the whole training + evaluation procedure or the only the evaluation part
ONLY_EVALUATION = False
# Number of images to print and save in evaluate.py
IMGS_NUMBER = 10
# Sparsity level. This parameter manages the sparsity level of OMP algorithm. Value should be in (0, 1) range.
# Recommended value: 0.25
SPARSE = 0.25
#  Number of K-SVD iterations
K_SVD_ITERATIONS = 20
# SEED
SEED = 1024



