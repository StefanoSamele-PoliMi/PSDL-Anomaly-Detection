import random
import numpy as np
import os
import pickle
from tqdm import tqdm
from sparselandtools.learning import ApproximateKSVD
from sparselandtools.pursuits import OrthogonalMatchingPursuit
from sparselandtools.dictionaries import DCTDictionary
from utils import build_model
import torch
from torch.utils.data import DataLoader
import time
import config
from math import floor, sqrt
from collections import OrderedDict
import torch.nn.functional as F


def train_single_layer():
    # device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load model
    model, layers_dictionary = build_model()

    model.to(device)
    model.eval()
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if use_cuda:
        torch.cuda.manual_seed_all(config.SEED)

    d = floor(sqrt(layers_dictionary[config.LAYER][1]))

    idx = torch.tensor(range(0, d*d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    layers_dictionary[config.LAYER][0].register_forward_hook(hook)

    os.makedirs(os.path.join(config.SAVE_PATH, 'arch_%s' % config.ARCH), exist_ok=True)

    # We can use different dataset and build a dataloader
    # train_dataset = mvtec_dataset.MVTecDataset(config.DATA_PATH, class_name='all', is_train=True)
    train_dataset = config.DATASET(config.DATA_PATH, class_name='all', is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)

    layer_outputs = []

    pixel_dicts = []

    # extract train set features
    train_class_dir = os.path.join(config.SAVE_PATH, 'arch_%s' % config.ARCH, 'all_classes')
    os.makedirs(train_class_dir, exist_ok=True)

    train_mean_filepath = os.path.join(train_class_dir, 'train_mean.pkl')
    train_cov_filepath = os.path.join(train_class_dir, 'train_cov.pkl')
    train_dict_filepath = os.path.join(train_class_dir, 'dictionaries.pkl')

    for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s | all_classes'):
        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        # get intermediate layer outputs

        layer_outputs.append(outputs[0].cpu().detach())

        # initialize hook outputs
        outputs = []

    embedding = torch.index_select(torch.cat(layer_outputs, 0), 1, idx)

    # Free up memory
    del(x)
    del(layer_outputs)

    # Calculate dictionary with K-SVD
    B, C, H, W = embedding.size()
    embedding_vectors = embedding.view(B, C, H * W)
    d1 = DCTDictionary(d, d)
    mean = np.zeros((2, H*W))
    cov = np.zeros((2, 2, H * W))
    I = np.identity(2)

    for i in range(H * W):

        # We apply approximate k-svd with 1/4 of dictionary dimension as sparsity level
        # 20 iterations applied.
        a_ksvd = ApproximateKSVD(d1, OrthogonalMatchingPursuit, round(d*d*config.SPARSE))
        start_time = time.time()
        learn_dict, coeff = a_ksvd.fit(embedding_vectors[:, :, i].transpose(1, 0).numpy(), config.K_SVD_ITERATIONS)
        print("--- %s seconds ---" % (time.time() - start_time))

        # We calculate the mean of l1 norm of each signal
        l1_norm = np.abs(coeff).sum(0)

        # We calculate the reconstruction error
        recon_err = (np.square(np.dot(learn_dict.matrix, coeff) - embedding_vectors[:, :, i].transpose(1, 0).numpy())).sum(axis=0)

        # We estimate the bivariate distibution of these two types of error
        mean[:, i] = np.array([np.mean(l1_norm), np.mean(recon_err)])
        cov[:, :, i] = np.cov(np.vstack([l1_norm, recon_err]).transpose(), rowvar=False) + 0.01 * I

        # We save the dictionary in a list
        pixel_dicts.append(learn_dict)

    with open(train_mean_filepath, 'wb') as f:
        pickle.dump(mean, f)

    with open(train_cov_filepath, 'wb') as f:
        pickle.dump(cov, f)

    with open(train_dict_filepath, 'wb') as f:
        pickle.dump(pixel_dicts, f)


def train_multiple_layers():
    # device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load model
    model, layers_dictionary = build_model()

    model.to(device)
    model.eval()
    random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if use_cuda:
        torch.cuda.manual_seed_all(config.SEED)

    d = 0
    for item in layers_dictionary.items():
        # Multiple layers model are made of only three layers
        if item[0] != 'layer4':
            d += item[1][1]

    dictionary_dim = floor(sqrt(d))
    surplus = d - (dictionary_dim*dictionary_dim)

    idx1 = torch.tensor(range(0, layers_dictionary['layer1'][1]-surplus))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    layers_dictionary['layer1'][0].register_forward_hook(hook)
    layers_dictionary['layer2'][0].register_forward_hook(hook)
    layers_dictionary['layer3'][0].register_forward_hook(hook)
    os.makedirs(os.path.join(config.SAVE_PATH, 'arch_%s' % config.ARCH), exist_ok=True)

    # We can use different dataset and build a dataloader
    train_dataset = config.DATASET(config.DATA_PATH, class_name='all', is_train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, pin_memory=True)

    layer_1_outputs = []
    layer_2_outputs = []
    layer_3_outputs = []

    pixel_dicts = []

    # extract train set features
    train_class_dir = os.path.join(config.SAVE_PATH, 'arch_%s' % config.ARCH, 'all_classes')
    os.makedirs(train_class_dir, exist_ok=True)

    train_mean_filepath = os.path.join(train_class_dir, 'train_mean.pkl')
    train_cov_filepath = os.path.join(train_class_dir, 'train_cov.pkl')
    train_dict_filepath = os.path.join(train_class_dir, 'dictionaries.pkl')

    for (x, y, mask) in tqdm(train_dataloader, '| feature extraction | train | %s | all_classes'):
        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        # get intermediate layer outputs

        layer_1_outputs.append(outputs[0].cpu().detach())
        layer_2_outputs.append(outputs[1].cpu().detach())
        layer_3_outputs.append(outputs[2].cpu().detach())
        # initialize hook outputs
        outputs = []

    layer_1_outputs = torch.index_select(torch.cat(layer_1_outputs, 0), 1, idx1)
    layer_2_outputs = torch.cat(layer_2_outputs, 0)
    layer_3_outputs = torch.cat(layer_3_outputs, 0)

    # Embedding concat
    embedding_vectors = layer_3_outputs
    if layer_2_outputs.size()[2] != embedding_vectors.size()[2]:
        embedding_vectors = torch.cat([embedding_vectors, F.avg_pool2d(layer_2_outputs, kernel_size=[2, 2])], dim=1)
        embedding_vectors = torch.cat([embedding_vectors, F.avg_pool2d(layer_1_outputs, kernel_size=[4, 4])], dim=1)

    else:
        embedding_vectors = torch.cat([embedding_vectors, layer_2_outputs], dim=1)
        embedding_vectors = torch.cat([embedding_vectors, layer_1_outputs], dim=1)

    # Free up memory
    del x
    del layer_1_outputs
    del layer_2_outputs
    del layer_3_outputs

    # Calculate dictionary with K-SVD
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    d1 = DCTDictionary(dictionary_dim, dictionary_dim)
    mean = np.zeros((2, H*W))
    cov = np.zeros((2, 2, H * W))
    I = np.identity(2)

    for i in range(H * W):

        # We apply approximate k-svd with 1/4 of dictionary dimension as sparsity level
        # 20 iterations applied.
        a_ksvd = ApproximateKSVD(d1, OrthogonalMatchingPursuit, round(dictionary_dim*dictionary_dim*config.SPARSE))
        start_time = time.time()
        learn_dict, coeff = a_ksvd.fit(embedding_vectors[:, :, i].transpose(1, 0).numpy(), config.K_SVD_ITERATIONS)
        print("--- %s seconds ---" % (time.time() - start_time))

        # We calculate the mean of l1 norm of each signal
        l1_norm = np.abs(coeff).sum(0)

        # We calculate the reconstruction error
        recon_err = (np.square(np.dot(learn_dict.matrix, coeff) - embedding_vectors[:, :, i].transpose(1, 0).numpy())).sum(axis=0)

        # We estimate the bivariate distibution of these two types of error
        mean[:, i] = np.array([np.mean(l1_norm), np.mean(recon_err)])
        cov[:, :, i] = np.cov(np.vstack([l1_norm, recon_err]).transpose(), rowvar=False) + 0.01 * I

        # We save the dictionary in a list
        pixel_dicts.append(learn_dict)

    with open(train_mean_filepath, 'wb') as f:
        pickle.dump(mean, f)

    with open(train_cov_filepath, 'wb') as f:
        pickle.dump(cov, f)

    with open(train_dict_filepath, 'wb') as f:
        pickle.dump(pixel_dicts, f)
