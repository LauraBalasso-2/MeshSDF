import glob
import os

import numpy as np
import trimesh

from lib import mesh_to_point_cloud, load_experiment_specifications, get_reconstruction_dir
from lib.IO_utils import create_dir_if_not_existent

import sys

sys.path.append('/home/laura/source/Github_repos/LauraBalasso-2/mesh_to_sdf/')
from mesh_to_sdf import create_from_scans

sys.path.append('/home/laura/source/mesh2sdf/')
from npz_to_vtu import save_vtu

def get_distance_function_error(true_mesh_path, reconstructed_mesh_path, save_path):
    mesh = trimesh.load(true_mesh_path)

    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("The mesh parameter must be a trimesh mesh.")

    sample_point_cloud = create_from_scans(mesh)
    point_cloud_translated = mesh_to_point_cloud(reconstructed_mesh_path)
    sdf = sample_point_cloud.get_sdf(point_cloud_translated)

    data = {'distance_from_true': np.ascontiguousarray(np.abs(sdf)),
            'x': np.ascontiguousarray(point_cloud_translated[:, 0]),
            'y': np.ascontiguousarray(point_cloud_translated[:, 1]),
            'z': np.ascontiguousarray(point_cloud_translated[:, 2])}

    save_vtu(save_path, data)


def compute_reconstruction_error(experiment_directory, validation=False):
    specs = load_experiment_specifications(experiment_directory)

    data_source = specs.get("DataSource")
    true_meshes_dir = os.path.join(data_source, 'rescaled_meshes')
    reconstruction_dir = get_reconstruction_dir(experiment_directory, validation=validation)
    reconstructed_meshes = glob.glob(os.path.join(reconstruction_dir, "body*"))
    if validation:
        label = 'validation'
    else:
        label = 'training'
    save_dir = os.path.join(experiment_directory, "model_validation", "reconstruction_error_" + label)
    create_dir_if_not_existent(save_dir)
    for reconstruction in reconstructed_meshes:
        sample_name = reconstruction.split('.')[0].split('/')[-1]
        true_mesh = os.path.join(true_meshes_dir, sample_name + '_rescaled.stl')
        save_path = os.path.join(save_dir, sample_name + "_error")
        get_distance_function_error(true_mesh, reconstruction, save_path)


def compute_clamped_error(predicted_sdf, true_sdf):
    true_np = true_sdf.numpy()
    pred_np = predicted_sdf.detach().numpy().reshape(-1)

    return np.abs(true_np[np.abs(true_np) < 0.1] - pred_np[np.abs(true_np) < 0.1])
