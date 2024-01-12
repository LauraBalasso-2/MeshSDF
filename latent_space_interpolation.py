import argparse

import numpy as np
import os
import sys

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

code_4 = latent_vectors[4]
code_7 = latent_vectors[7]

code_mean = torch.mean(torch.stack([code_4, code_7]), dim=0)
code_025 = torch.mean(torch.stack([code_4, code_mean]), dim=0)
code_075 = torch.mean(torch.stack([code_7, code_mean]), dim=0)
# code_075 = np.mean([code_mean, code_7.numpy().reshape(-1, 1)], axis=0)


# code_mean = np.mean([code_4.numpy().reshape(-1, 1), code_7.numpy().reshape(-1, 1)], axis=0)
#
# code_025 = np.mean([code_4.numpy().reshape(-1, 1), code_mean], axis=0)
# code_075 = np.mean([code_mean, code_7.numpy().reshape(-1, 1)], axis=0)

decoder, epoch = load_decoder(experiment_directory, 'latest')
decoder.eval()

intermediate_codes = [code_025, code_mean, code_075]
file_names = ['']

x = np.random.uniform(-1, 1, size=250000)
y = np.random.uniform(-0.5, 0.5, size=250000)
z = np.random.uniform(-0.3, 0.3, size=250000)
xyz = torch.as_tensor(np.column_stack([x, y, z]), dtype=torch.float)
save_dir = os.path.join(experiment_directory, 'latent_space_interpolation')
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

for i, code in enumerate(intermediate_codes):
    predicted_sdf = decode_sdf(decoder, code, xyz)
    data_pred = {'sdf_predicted': np.ascontiguousarray(predicted_sdf.detach().numpy().reshape(-1)),
                 'x': np.ascontiguousarray(xyz[:, 0]),
                 'y': np.ascontiguousarray(xyz[:, 1]),
                 'z': np.ascontiguousarray(xyz[:, 2])}
    save_vtu(os.path.join(save_dir, 'predicted_sdf_'+str(i).zfill(2)), data_pred)

    reconstruct_mesh(decoder, 'reconstruction_'+str(i).zfill(2), code, save_dir)