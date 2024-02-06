import torch
import numpy as np
import json
import os
from lib import load_latent_vectors, load_decoder, decode_sdf, load_experiment_specifications, get_instance_filenames, \
    unpack_sdf_samples, get_reconstruction_dir, create_mesh, get_mesh_filename, \
    Logger
from lib.error_computation import compute_clamped_error
from lib.IO_utils import create_dir_if_not_existent

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
sys.path.append('/home/laura/source/utilities')
from error_plots import plot_error_hist, plot_error_bar, plot_multiple_boxplots
sys.path.append('/home/laura/source/mesh2sdf/')
from npz_to_vtu import save_vtu

def optimize_latent_vector(decoder, num_iterations, latent_size, test_sdf, stat, clamp_dist, num_samples, lr, l2reg,
                           logger):
    def adjust_learning_rate(
            initial_lr, optim, num_iter, decreased_by_m, adjust_lr_every_m
    ):
        l_rate = initial_lr * ((1 / decreased_by_m) ** (num_iter // adjust_lr_every_m))
        for param_group in optim.param_groups:
            param_group["lr"] = l_rate

    decreased_by = 10
    adjust_lr_every = int(num_iterations * 2 / 3)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat)
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach())
    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()

    logger.write('Starting latent code optimization ...')

    for e in range(num_iterations):
        decoder.eval()
        xyz = test_sdf[:, 0:3]
        sdf_gt = test_sdf[:, 3].unsqueeze(1)
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)
        # inputs = torch.cat([latent_inputs, xyz], 1)

        pred_sdf = decoder(latent_inputs, xyz)
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 10 == 0:
            logger.write('iter ' + str(e), level='debug')
            logger.write('loss ' + str(loss.cpu().data.numpy()) + '\n', level='debug')
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


def predict_sdf(experiment_directory, save_true=True, save_predicted=True, validation=False, mesh_reconstruction=False,
                sample_error_plot=True, error_summary_plot='bar', logger=Logger()):

    specs = load_experiment_specifications(experiment_directory)
    data_source = specs["DataSource"]
    num_samp_per_scene = specs["SamplesPerScene"]
    if validation:
        split_file = specs['TestSplit']
        label = "validation"
    else:
        split_file = specs["TrainSplit"]
        label = "training"

    with open(split_file, "r") as f:
        train_split = json.load(f)

    save_error_dir, save_true_dir, save_predicted_dir, img_dir = setup_directories(experiment_directory,
                                                                                   save_predicted,
                                                                                   save_true,
                                                                                   label=label)
    reconstruction_dir = get_reconstruction_dir(experiment_directory, True, validation=validation)

    npz_files = get_instance_filenames(data_source, train_split)

    decoder, epoch = load_decoder(experiment_directory, 'latest')
    decoder.eval()

    if not validation:
        latent_vectors = load_latent_vectors(experiment_directory, checkpoint='latest')

    error_summary = {}

    for i, file in enumerate(sorted(npz_files)):
        logger.write("Predicting sample from file: " + file)
        file_path = os.path.join(data_source, file)
        sample_name = file.split('.')[0].split('/')[-1]
        subsample = unpack_sdf_samples(file_path, subsample=num_samp_per_scene, random_seed=i)
        xyz = subsample[:, :-1]
        true_sdf = subsample[:, -1]
        if validation:
            _, latent_code = optimize_latent_vector(decoder,
                                                    300,
                                                    specs["CodeLength"],
                                                    subsample,
                                                    stat=0.01,
                                                    clamp_dist=0.1,
                                                    num_samples=num_samp_per_scene,
                                                    lr=5e-3,
                                                    l2reg=True,
                                                    logger=logger)
        else:
            latent_code = latent_vectors[i]
        predicted_sdf = decode_sdf(decoder, latent_code, xyz)
        sdf_error = compute_clamped_error(predicted_sdf, true_sdf)
        if error_summary_plot == 'box_plots':
            error_summary[sample_name] = sdf_error.astype(float).tolist()
        else:
            error_summary[sample_name] = float(np.mean(sdf_error))
        if sample_error_plot:
            plot_error_hist(sdf_error, fig_name=os.path.join(img_dir, sample_name + '_error_hist.png'), bins=20)
        clamped_coordinates = xyz[np.abs(true_sdf.numpy()) < 0.1]
        save_3d_function(clamped_coordinates, sdf_error, os.path.join(save_error_dir, sample_name), function_name='sdf_error')
        if save_true:
            save_3d_function(xyz, true_sdf, os.path.join(save_true_dir, sample_name), function_name='sdf')
        if save_predicted:
            save_3d_function(xyz, predicted_sdf.detach().numpy().reshape(-1), os.path.join(save_predicted_dir, sample_name), function_name='sdf_predicted')

        if mesh_reconstruction:
            reconstruct_mesh(decoder, file, latent_code, reconstruction_dir)

    plot_error(error_summary, error_summary_plot, img_dir, save_error_dir)


def save_3d_function(coordinates, function_values, file_path, function_name):
    data = {function_name: np.ascontiguousarray(function_values),
            'x': np.ascontiguousarray(coordinates[:, 0]),
            'y': np.ascontiguousarray(coordinates[:, 1]),
            'z': np.ascontiguousarray(coordinates[:, 2])}
    save_vtu(file_path, data)


def plot_error(error_summary, error_summary_plot, img_dir, save_error_dir):
    if error_summary_plot == 'bar':
        plot_error_bar(list(error_summary.values()), labels=list(error_summary.keys()), plot_mean=True,
                       fig_name=os.path.join(img_dir, 'mean_errors.png'))

    elif error_summary_plot == 'hist':
        plot_error_hist(list(error_summary.values()), fig_name=os.path.join(img_dir, 'mean_errors.png'))

    elif error_summary_plot == 'box_plots':
        plot_multiple_boxplots(error_summary, fig_name=os.path.join(img_dir, 'error_box_plots.png'))
        for k in error_summary.keys():
            error_summary[k] = float(np.mean(error_summary[k]))
    with open(os.path.join(save_error_dir, 'mean_errors.json'), 'w') as f:
        json.dump(error_summary, f)


def reconstruct_mesh(decoder, file, latent_code, reconstruction_dir, resolution=512):
    mesh_filename = get_mesh_filename(reconstruction_dir, file)
    print("Reconstructing {}...".format(mesh_filename))
    with torch.no_grad():
        create_mesh(decoder, latent_code, N=resolution, output_mesh=False, filename=mesh_filename)


def setup_directories(experiment_directory, save_predicted, save_true, label):
    save_dir = os.path.join(experiment_directory, 'model_validation')
    create_dir_if_not_existent(save_dir)
    if save_true:
        save_true_dir = os.path.join(experiment_directory, 'sdf_' + label + '_subsamples')
        create_dir_if_not_existent(save_true_dir)
    else:
        save_true_dir = None

    if save_predicted:
        save_predicted_dir = os.path.join(save_dir, 'sdf_' + label + '_predicted_subsamples')
        create_dir_if_not_existent(save_predicted_dir)
    else:
        save_predicted_dir = None

    save_error_dir = os.path.join(save_dir, 'sdf_' + label + '_prediction_error')
    create_dir_if_not_existent(save_error_dir)
    img_dir = os.path.join(save_dir, 'img_'+label)
    create_dir_if_not_existent(img_dir)

    return save_error_dir, save_true_dir, save_predicted_dir, img_dir

