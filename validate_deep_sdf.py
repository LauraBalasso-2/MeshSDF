import os.path

from lib import predict_sdf, Logger
from lib.error_computation import compute_reconstruction_error
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

logger_path = os.path.join(args.experiment_directory, 'sdf_prediction.log')

# predict_sdf(args.experiment_directory, mesh_reconstruction=False, sample_error_plot=False, error_summary_plot='hist')

# predict_sdf(args.experiment_directory,
#             validation=True,
#             mesh_reconstruction=True,
#             sample_error_plot=True,
#             error_summary_plot='bar',
#             logger=Logger(logger_path, level='debug'))

# compute_reconstruction_error(args.experiment_directory)

#  compute_reconstruction_error(args.experiment_directory, validation=True)
