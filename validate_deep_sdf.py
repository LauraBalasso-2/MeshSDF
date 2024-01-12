from lib import predict_sdf, compute_reconstruction_error
import argparse

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



predict_sdf(args.experiment_directory, mesh_reconstruction=True)

# predict_sdf(args.experiment_directory, validation=True, mesh_reconstruction=True)

# compute_reconstruction_error(args.experiment_directory)

# compute_reconstruction_error(args.experiment_directory, validation=True)
