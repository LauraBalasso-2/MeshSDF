import trimesh
import open3d as o3d
import numpy as np
import os

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

    data = {'sdf': np.ascontiguousarray(np.abs(sdf)),
            'x': np.ascontiguousarray(point_cloud_translated[:, 0]),
            'y': np.ascontiguousarray(point_cloud_translated[:, 1]),
            'z': np.ascontiguousarray(point_cloud_translated[:, 2])}

    save_vtu(save_path, data)


