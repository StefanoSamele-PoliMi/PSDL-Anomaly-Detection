import config
import numpy as np
import os
import pickle
from tqdm import tqdm
from utils import build_model
from math import floor, sqrt
from sparselandtools.pursuits import OrthogonalMatchingPursuit
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import roc_curve
from utils import plot_fig

# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def evaluate_single_layer():
    # device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load model
    model, layers_dictionary = build_model()


    model.to(device)
    model.eval()
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if use_cuda:
        torch.cuda.manual_seed_all(config.SEED)

    d = floor(sqrt(layers_dictionary[config.LAYER][1]))

    idx = torch.tensor(range(0, d * d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    layers_dictionary[config.LAYER][0].register_forward_hook(hook)

    arch_filepath = os.path.join(config.SAVE_PATH, 'arch_%s' % config.ARCH)

    train_class_dir = os.path.join(arch_filepath, 'all_classes')
    train_mean_filepath = os.path.join(train_class_dir, 'train_mean.pkl')
    train_cov_filepath = os.path.join(train_class_dir, 'train_cov.pkl')
    train_dict_filepath = os.path.join(train_class_dir, 'dictionaries.pkl')

    image_path = os.path.join(arch_filepath, 'images')
    os.makedirs(image_path, exist_ok=True)

    with open(train_mean_filepath, 'rb') as f:
        mean = pickle.load(f)

    with open(train_cov_filepath, 'rb') as f:
        cov = pickle.load(f)

    with open(train_dict_filepath, 'rb') as f:
        dict = pickle.load(f)

    total_img_roc_auc = []
    total_pixel_roc_auc = []
    total_img_pr_auc = []
    total_pixel_pr_auc = []

    for class_name in config.CLASS_NAMES:

        test_dataset = config.DATASET(config.DATA_PATH, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        layer_outputs = []

        test_imgs = []
        gt_list = []
        gt_mask_list = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs

            layer_outputs.append(outputs[0].cpu().detach())

            # initialize hook outputs
            outputs = []

        embedding = torch.index_select(torch.cat(layer_outputs, 0), 1, idx)

        # Free up memory
        del (layer_outputs)

        # calculate distance matrix
        B, C, H, W = embedding.size()
        embedding_vectors = embedding.view(B, C, H * W).detach().numpy()
        dist_list = []

        for i in range(H*W):
            sparse_rep = OrthogonalMatchingPursuit(dict[i], sparsity=round(d*d*config.SPARSE)).fit(embedding_vectors[:, :, i].transpose(1,0))

            # We calculate the mean of l1 norm of each signal
            l1_norm = np.abs(sparse_rep).sum(0)

            # We calculate the reconstruction error
            recon_err = (np.square(np.dot(dict[i].matrix, sparse_rep) - embedding_vectors[:, :, i].transpose(1, 0))).sum(
                axis=0)

            conv_inv_pixel = np.linalg.inv(cov[:, :, i])
            samples = np.vstack([l1_norm, recon_err]).transpose()

            dist = [mahalanobis(sample, mean[:, i], conv_inv_pixel) for sample in samples]
            dist_list.append(dist)

        score_map_mini = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        score_map = F.interpolate(torch.tensor(np.expand_dims(score_map_mini, 1)), size=x.size()[-2:], mode='bicubic',
                                  align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
           score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        # calculate image-level ROC AUC score
        img_scores = scores.max(axis=(1, 2))
        labels = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(labels, img_scores)
        total_img_roc_auc.append(img_roc_auc)
        print('%s image ROCAUC: %.3f' % (class_name, img_roc_auc))
        img_pr_auc = average_precision_score(labels, img_scores)
        total_img_pr_auc.append(img_pr_auc)
        print('%s image PR-AUC: %.3f' % (class_name, img_pr_auc))

        gt_mask = np.asarray(gt_mask_list)
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print( '%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        per_pixel_pr_auc = average_precision_score(gt_mask.flatten(), scores.flatten())
        total_pixel_pr_auc.append(per_pixel_pr_auc)
        print('%s pixel PR-AUC: %.3f' % (class_name, per_pixel_pr_auc))

        # get optimal threshold
        fpr, tpr, thresholds = roc_curve(gt_mask.flatten(), scores.flatten())
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Save a subset of images
        image_classs_path = os.path.join(image_path, '%s' % class_name)
        os.makedirs(image_classs_path, exist_ok=True)
        images_idx = np.random.randint(0, B, config.IMGS_NUMBER)

        plot_fig(np.asarray(test_imgs)[images_idx], scores[images_idx], gt_mask[images_idx], optimal_threshold,
                 image_classs_path)

    print('Average image ROCAUC: %.3f' % np.mean(total_img_roc_auc))
    print('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    print('Average image PRAUC: %.3f' % np.mean(total_img_pr_auc))
    print('Average pixel PRAUC: %.3f' % np.mean(total_pixel_pr_auc))


def evaluate_multiple_layers():
    # device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # load model
    model, layers_dictionary = build_model()

    model.to(device)
    model.eval()
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    if use_cuda:
        torch.cuda.manual_seed_all(config.SEED)

    d = 0
    for item in layers_dictionary.items():
        # Multiple layers model are made of only three layers
        if item[0] != 'layer4':
            d += item[1][1]

    dictionary_dim = floor(sqrt(d))
    surplus = d - (dictionary_dim * dictionary_dim)

    idx1 = torch.tensor(range(0, layers_dictionary['layer1'][1] - surplus))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    layers_dictionary['layer1'][0].register_forward_hook(hook)
    layers_dictionary['layer2'][0].register_forward_hook(hook)
    layers_dictionary['layer3'][0].register_forward_hook(hook)

    # set model's intermediate outputs
    outputs = []

    arch_filepath = os.path.join(config.SAVE_PATH, 'arch_%s' % config.ARCH)

    train_class_dir = os.path.join(arch_filepath, 'all_classes')
    train_mean_filepath = os.path.join(train_class_dir, 'train_mean.pkl')
    train_cov_filepath = os.path.join(train_class_dir, 'train_cov.pkl')
    train_dict_filepath = os.path.join(train_class_dir, 'dictionaries.pkl')

    image_path = os.path.join(arch_filepath, 'images')
    os.makedirs(image_path, exist_ok=True)

    with open(train_mean_filepath, 'rb') as f:
        mean = pickle.load(f)

    with open(train_cov_filepath, 'rb') as f:
        cov = pickle.load(f)

    with open(train_dict_filepath, 'rb') as f:
        dict = pickle.load(f)

    total_img_roc_auc = []
    total_pixel_roc_auc = []
    total_img_pr_auc = []
    total_pixel_pr_auc = []

    for class_name in config.CLASS_NAMES:

        test_dataset = config.DATASET(config.DATA_PATH, class_name=class_name, is_train=False)
        test_dataloader = DataLoader(test_dataset, batch_size=32, pin_memory=True)

        layer_1_outputs = []
        layer_2_outputs = []
        layer_3_outputs = []

        test_imgs = []
        gt_list = []
        gt_mask_list = []

        # extract test set features
        for (x, y, mask) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
            test_imgs.extend(x.cpu().detach().numpy())
            gt_list.extend(y.cpu().detach().numpy())
            gt_mask_list.extend(mask.cpu().detach().numpy())
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
            embedding_vectors = torch.cat([embedding_vectors, layer_2_outputs])
            embedding_vectors = torch.cat([embedding_vectors, layer_1_outputs])

        # Free up memory
        del layer_1_outputs
        del layer_2_outputs
        del layer_3_outputs

        # calculate distance matrix
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).detach().numpy()
        dist_list = []

        for i in range(H * W):
            sparse_rep = OrthogonalMatchingPursuit(dict[i], sparsity=round(dictionary_dim*dictionary_dim*config.SPARSE)).fit(
                embedding_vectors[:, :, i].transpose(1, 0))

            # We calculate the mean of l1 norm of each signal
            l1_norm = np.abs(sparse_rep).sum(0)

            # We calculate the reconstruction error
            recon_err = (
                np.square(np.dot(dict[i].matrix, sparse_rep) - embedding_vectors[:, :, i].transpose(1, 0))).sum(
                axis=0)

            conv_inv_pixel = np.linalg.inv(cov[:, :, i])
            samples = np.vstack([l1_norm, recon_err]).transpose()

            dist = [mahalanobis(sample, mean[:, i], conv_inv_pixel) for sample in samples]
            dist_list.append(dist)

        score_map_mini = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

        # upsample
        score_map = F.interpolate(torch.tensor(np.expand_dims(score_map_mini, 1)), size=x.size()[-2:], mode='bicubic',
                                  align_corners=False).squeeze().numpy()

        # apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)

        # Normalization
        max_score = score_map.max()
        min_score = score_map.min()
        scores = (score_map - min_score) / (max_score - min_score)

        # calculate image-level ROC AUC score
        img_scores = scores.max(axis=(1, 2))
        labels = np.asarray(gt_list)
        img_roc_auc = roc_auc_score(labels, img_scores)
        total_img_roc_auc.append(img_roc_auc)
        print('%s image ROCAUC: %.3f' % (class_name, img_roc_auc))
        img_pr_auc = average_precision_score(labels, img_scores)
        total_img_pr_auc.append(img_pr_auc)
        print('%s image PR-AUC: %.3f' % (class_name, img_pr_auc))

        gt_mask = np.asarray(gt_mask_list)
        per_pixel_rocauc = roc_auc_score(gt_mask.flatten(), scores.flatten())
        total_pixel_roc_auc.append(per_pixel_rocauc)
        print('%s pixel ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        per_pixel_pr_auc = average_precision_score(gt_mask.flatten(), scores.flatten())
        total_pixel_pr_auc.append(per_pixel_pr_auc)
        print('%s pixel PR-AUC: %.3f' % (class_name, per_pixel_pr_auc))

        # get optimal threshold
        fpr, tpr, thresholds = roc_curve(gt_mask.flatten(), scores.flatten())
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # Save a subset of images
        image_classs_path = os.path.join(image_path, '%s' % class_name)
        os.makedirs(image_classs_path, exist_ok=True)
        images_idx = np.random.randint(0, B, config.IMGS_NUMBER)

        plot_fig(np.asarray(test_imgs)[images_idx], scores[images_idx], gt_mask[images_idx], optimal_threshold,
                 image_classs_path)

    print('Average image ROCAUC: %.3f' % np.mean(total_img_roc_auc))
    print('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    print('Average image PRAUC: %.3f' % np.mean(total_img_pr_auc))
    print('Average pixel PRAUC: %.3f' % np.mean(total_pixel_pr_auc))
