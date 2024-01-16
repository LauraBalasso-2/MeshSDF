import glob

import torch
import trimesh
import open3d as o3d
import numpy as np
import json
import os
import logging
from lib import load_latent_vectors, load_decoder, decode_sdf, load_experiment_specifications, SDFSamples, \
    get_instance_filenames, unpack_sdf_samples, get_reconstruction_dir, create_mesh, get_mesh_filename

os.environ['PYOPENGL_PLATFORM'] = 'egl'
import sys

sys.path.append('/home/laura/source/Github_repos/LauraBalasso-2/mesh_to_sdf/')
from mesh_to_sdf import create_from_scans

sys.path.append('/home/laura/source/mesh2sdf/')
from npz_to_vtu import save_vtu


def mesh_to_point_cloud(mesh_path):
    pcd = o3d.io.read_point_cloud(mesh_path)
    reconstructed_point_cloud = np.asarray(pcd.points)
    translation_matrix = np.ones_like(reconstructed_point_cloud)
    translated_point_cloud = reconstructed_point_cloud - translation_matrix
    return translated_point_cloud


def get_distance_function_error(true_mesh_path, reconstructed_mesh_path, save_path):
    mesh = trimesh.load(true_mesh_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")

    sample_point_cloud = create_from_scans(mesh)
    point_cloud_translated = mesh_to_point_cloud(reconstructed_mesh_path)
    sdf = sample_point_cloud.get_sdf(point_cloud_translated)

    data = {'distance_function': np.ascontiguousarray(np.abs(sdf)),
            'x': np.ascontiguousarray(point_cloud_translated[:, 0]),
            'y': np.ascontiguousarray(point_cloud_translated[:, 1]),
            'z': np.ascontiguousarray(point_cloud_translated[:, 2])}

    save_vtu(save_path, data)


def optimize_latent_vector(decoder, num_iterations, latent_size, test_sdf, stat, clamp_dist, num_samples, lr, l2reg):
    def adjust_learning_rate(
            initial_lr, optimizer, num_iter, decreased_by_m, adjust_lr_every_m
    ):
        l_rate = initial_lr * ((1 / decreased_by_m) ** (num_iter // adjust_lr_every_m))
        for param_group in optimizer.param_groups:
            param_group["lr"] = l_rate

    decreased_by = 10
    adjust_lr_every = int(num_iterations / 2)

    if type(stat) == type(0.1):
        latent = torch.ones(1, latent_size).normal_(mean=0, std=stat)
    else:
        latent = torch.normal(stat[0].detach(), stat[1].detach())
    latent.requires_grad = True

    optimizer = torch.optim.Adam([latent], lr=lr)

    loss_num = 0
    loss_l1 = torch.nn.L1Loss()
    for e in range(num_iterations):
        decoder.eval()
        xyz = test_sdf[:, 0:3]
        sdf_gt = test_sdf[:, 3].unsqueeze(1)
        sdf_gt = torch.clamp(sdf_gt, -clamp_dist, clamp_dist)
        adjust_learning_rate(lr, optimizer, e, decreased_by, adjust_lr_every)

        optimizer.zero_grad()

        latent_inputs = latent.expand(num_samples, -1)
        inputs = torch.cat([latent_inputs, xyz], 1)

        pred_sdf = decoder(inputs)
        pred_sdf = torch.clamp(pred_sdf, -clamp_dist, clamp_dist)
        loss = loss_l1(pred_sdf, sdf_gt)
        if l2reg:
            loss += 1e-4 * torch.mean(latent.pow(2))
        loss.backward()
        optimizer.step()

        if e % 50 == 0:
            logging.debug(loss.Fcpu().data.numpy())
            logging.debug(e)
            logging.debug(latent.norm())
        loss_num = loss.cpu().data.numpy()

    return loss_num, latent


def predict_sdf(experiment_directory, save_true=True, save_predicted=True, validation=False, mesh_reconstruction=False):
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

    save_error_dir, save_true_dir, save_predicted_dir = setup_directories(experiment_directory,
                                                                          save_predicted,
                                                                          save_true,
                                                                          label=label)
    reconstruction_dir = get_reconstruction_dir(experiment_directory, True)

    npz_files = get_instance_filenames(data_source, train_split)

    decoder, epoch = load_decoder(experiment_directory, 'latest')
    decoder.eval()

    if not validation:
        latent_vectors = load_latent_vectors(experiment_directory, checkpoint='latest')

    for i, file in enumerate(sorted(npz_files)):
        file_path = os.path.join(data_source, file)
        subsample = unpack_sdf_samples(file_path, subsample=num_samp_per_scene, random_seed=i)
        xyz = subsample[:, :-1]
        true_sdf = subsample[:, -1]
        if validation:
            _, latent_code = optimize_latent_vector(decoder,
                                                    800,
                                                    specs["CodeLength"],
                                                    subsample,
                                                    stat=0.01,
                                                    clamp_dist=0.1,
                                                    num_samples=num_samp_per_scene,
                                                    lr=1e-5,
                                                    l2reg=True)
        else:
            latent_code = latent_vectors[i]
        predicted_sdf = decode_sdf(decoder, latent_code, xyz)
        sdf_error = np.abs(true_sdf.numpy() - predicted_sdf.detach().numpy().reshape(-1))
        data = {'sdf_error': np.ascontiguousarray(sdf_error),
                'x': np.ascontiguousarray(xyz[:, 0]),
                'y': np.ascontiguousarray(xyz[:, 1]),
                'z': np.ascontiguousarray(xyz[:, 2])}
        save_vtu(os.path.join(save_error_dir, file.split('.')[0].split('/')[-1]), data)
        if save_true:
            data_true = {'sdf': np.ascontiguousarray(true_sdf),
                         'x': np.ascontiguousarray(xyz[:, 0]),
                         'y': np.ascontiguousarray(xyz[:, 1]),
                         'z': np.ascontiguousarray(xyz[:, 2])}
            save_vtu(os.path.join(save_true_dir, file.split('.')[0].split('/')[-1]), data_true)
        if save_predicted:
            data_pred = {'sdf_predicted': np.ascontiguousarray(predicted_sdf.detach().numpy().reshape(-1)),
                         'x': np.ascontiguousarray(xyz[:, 0]),
                         'y': np.ascontiguousarray(xyz[:, 1]),
                         'z': np.ascontiguousarray(xyz[:, 2])}
            save_vtu(os.path.join(save_predicted_dir, file.split('.')[0].split('/')[-1]), data_pred)

        if mesh_reconstruction:
            reconstruct_mesh(decoder, file, latent_code, reconstruction_dir)


def reconstruct_mesh(decoder, file, latent_code, reconstruction_dir, resolution=256):
    mesh_filename = get_mesh_filename(reconstruction_dir, file)
    print("Reconstructing {}...".format(mesh_filename))
    with torch.no_grad():
        create_mesh(decoder, latent_code, N=resolution, output_mesh=False, filename=mesh_filename)


def setup_directories(experiment_directory, save_predicted, save_true, label):
    save_dir = os.path.join(experiment_directory, 'model_validation')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if save_true:
        save_true_dir = os.path.join(experiment_directory, 'sdf_' + label + '_subsamples')
        if not os.path.isdir(save_true_dir):
            os.mkdir(save_true_dir)
    else:
        save_true_dir = None

    if save_predicted:
        save_predicted_dir = os.path.join(save_dir, 'sdf_' + label + '_predicted_subsamples')
        if not os.path.isdir(save_predicted_dir):
            os.mkdir(save_predicted_dir)
    else:
        save_predicted_dir = None

    save_error_dir = os.path.join(save_dir, 'sdf_' + label + '_prediction_error')
    if not os.path.isdir(save_error_dir):
        os.mkdir(save_error_dir)

    return save_error_dir, save_true_dir, save_predicted_dir


def compute_reconstruction_error(experiment_directory, validation=False):
    # TODO : compute error only for train / validation instances or remove the option

    specs = load_experiment_specifications(experiment_directory)
    if validation:
        label = "validation"
        split = specs.get('TestSplit')
    else:
        label = "training"
        split = specs.get("TrainSplit")

    data_source = specs.get("DataSource")
    true_meshes_dir = os.path.join(data_source, 'rescaled_meshes')
    reconstructed_meshes = glob.glob(os.path.join(experiment_directory, "Reconstructions/body*"))
    print(reconstructed_meshes)
    save_dir = os.path.join(experiment_directory, "model_validation", "reconstructions_sdf_" + label)
    if not os.path.isdir(save_dir):
        os.system('mkdir -p ' + save_dir)
    for reconstruction in reconstructed_meshes:
        print(reconstruction)
        sample_name = reconstruction.split('.')[0].split('/')[-1]
        true_mesh = os.path.join(true_meshes_dir, sample_name + '_rescaled.stl')
        save_path = os.path.join(save_dir, sample_name + "_sdf_error")
        get_distance_function_error(true_mesh, reconstruction, save_path)
