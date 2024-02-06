import argparse

import numpy as np
import os
import json
import sys

import matplotlib.pyplot as plt

import torch

sys.path.append('/home/laura/source/mesh2sdf/')
from npz_to_vtu import save_vtu
from lib import load_latent_vectors, load_decoder, decode_sdf, reconstruct_mesh

arg_parser = argparse.ArgumentParser(description="Compute latent space interpolations")
arg_parser.add_argument(
    "--experiment",
    "-e",
    dest="experiment_directory",
    required=True,
    help="The experiment directory. This directory should include experiment "
         + "specifications in 'specs.json', and logging will be done in this directory "
         + "as well",
)
args = arg_parser.parse_args()

experiment_directory = args.experiment_directory
latent_vectors = load_latent_vectors(experiment_directory, checkpoint='latest')

np_latent = latent_vectors.numpy()


with open('/home/laura/exclude_backup/pycharm_projects_deployment/MeshSDF/splits/drivaer_2var_randomSplit_train.json', 'r') as f:
    split = json.load(f)

train_samples = split.get('sdf_samples').get('npz')

plt.scatter(np_latent[:, 0], np_latent[:, 1])
i_min = np.argmin(np_latent[:, 0])
i_max = np.argmax(np_latent[:, 0])

for i, t in enumerate(sorted(train_samples)):
    plt.annotate(str(int(t[-3:])), (np_latent[i, 0], np_latent[i, 1]))
plt.scatter([np_latent[i_min, 0], np_latent[i_max, 0]], [np_latent[i_min, 1], np_latent[i_max, 1]], color='red')
plt.savefig(os.path.join(experiment_directory, 'latent_space_2.png'))

# code_min = latent_vectors[i_min]
# code_max = latent_vectors[i_max]
#
# code_mean = torch.mean(torch.stack([code_min, code_max]), dim=0)
# code_025 = torch.mean(torch.stack([code_min, code_mean]), dim=0)
# code_075 = torch.mean(torch.stack([code_max, code_mean]), dim=0)
#
#
# decoder, epoch = load_decoder(experiment_directory, 'latest')
# decoder.eval()
#
# intermediate_codes = [code_025, code_mean, code_075]
# file_names = ['']
#
# x = np.random.uniform(-1, 1, size=250000)
# y = np.random.uniform(-0.5, 0.5, size=250000)
# z = np.random.uniform(-0.3, 0.3, size=250000)
# xyz = torch.as_tensor(np.column_stack([x, y, z]), dtype=torch.float)
# save_dir = os.path.join(experiment_directory, 'latent_space_interpolation')
# if not os.path.isdir(save_dir):
#     os.mkdir(save_dir)
#
# for i, code in enumerate(intermediate_codes):
#     predicted_sdf = decode_sdf(decoder, code, xyz)
#     data_pred = {'sdf_predicted': np.ascontiguousarray(predicted_sdf.detach().numpy().reshape(-1)),
#                  'x': np.ascontiguousarray(xyz[:, 0]),
#                  'y': np.ascontiguousarray(xyz[:, 1]),
#                  'z': np.ascontiguousarray(xyz[:, 2])}
#     save_vtu(os.path.join(save_dir, 'predicted_sdf_'+str(i).zfill(2)), data_pred)
#
#     reconstruct_mesh(decoder, 'reconstruction_'+str(i).zfill(2), code, save_dir)