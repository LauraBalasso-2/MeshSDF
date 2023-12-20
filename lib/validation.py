import trimesh
import open3d as o3d
import numpy as np
import json
import os
from lib import load_latent_vectors, load_decoder, decode_sdf, load_experiment_specifications, SDFSamples, \
    get_instance_filenames, unpack_sdf_samples

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


def predict_sdf_training(experiment_directory, save_true=True, save_predicted=True):
    save_error_dir, save_true_dir, save_predicted_dir = setup_directories(experiment_directory, save_predicted, save_true)

    specs = load_experiment_specifications(experiment_directory)
    data_source = specs["DataSource"]
    num_samp_per_scene = specs["SamplesPerScene"]
    train_split_file = specs["TrainSplit"]

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    npz_files = get_instance_filenames(data_source, train_split)

    latent_vectors = load_latent_vectors(experiment_directory, checkpoint='latest')

    decoder, epoch = load_decoder(experiment_directory, 'latest')
    decoder.eval()


    for i, file in enumerate(npz_files):
        file_path = os.path.join(data_source, file)
        subsample = unpack_sdf_samples(file_path, subsample=num_samp_per_scene, random_seed=i)
        xyz = subsample[:, :-1]
        true_sdf = subsample[:, -1]
        predicted_sdf = decode_sdf(decoder, latent_vectors[i], xyz)
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


def setup_directories(experiment_directory, save_predicted, save_true):
    save_dir = os.path.join(experiment_directory, 'model_validation')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if save_true:
        save_true_dir = os.path.join(experiment_directory, 'sdf_training_subsamples')
        if not os.path.isdir(save_true_dir):
            os.mkdir(save_true_dir)
    else:
        save_true_dir = None

    if save_predicted:
        save_predicted_dir = os.path.join(save_dir, 'sdf_predicted_subsamples')
        if not os.path.isdir(save_predicted_dir):
            os.mkdir(save_predicted_dir)
    else:
        save_predicted_dir = None

    save_error_dir = os.path.join(save_dir, 'sdf_prediction_error')
    if not os.path.isdir(save_error_dir):
        os.mkdir(save_error_dir)

    return save_error_dir, save_true_dir, save_predicted_dir

