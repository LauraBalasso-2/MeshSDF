from lib import get_distance_function_error
from lib import load_experiment_specifications
import argparse
import glob
import os

arg_parser = argparse.ArgumentParser(description="Compute validation errors")
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

specs = load_experiment_specifications(args.experiment_directory)
data_source = specs.get("DataSource")
true_meshes_dir = os.path.join("/", *data_source.split("/")[:-1], 'rescaled_meshes')

reconstructed_meshes = glob.glob(os.path.join(args.experiment_directory, "Reconstructions/body*"))
print(reconstructed_meshes)

save_dir = os.path.join(args.experiment_directory, "model_validation", "reconstructions_sdf")

for reconstruction in reconstructed_meshes:
    print(reconstruction)
    sample_name = reconstruction.split('.')[0].split('/')[-1]
    true_mesh = os.path.join(true_meshes_dir, sample_name + '_rescaled.stl')
    save_path = os.path.join(save_dir, sample_name + "_sdf_error")
    get_distance_function_error(true_mesh, reconstruction, save_path)
