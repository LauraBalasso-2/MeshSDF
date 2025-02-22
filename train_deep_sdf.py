#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import torch.utils.data as data_utils
import signal
import sys
import math

import lib
from lib.workspace import *
from lib.models.decoder import *
from lib.mesh import *


def main_function(experiment_directory, resolution):
    specs = load_experiment_specifications(experiment_directory)

    print("Experiment description: \n" + ' '.join([str(elem) for elem in specs["Description"]]))

    data_source = specs["DataSource"]
    train_split_file = specs["TrainSplit"]
    latent_size = specs["CodeLength"]

    lr_schedules = get_learning_rate_schedules(specs)

    def save_latest(epoch):
        save_model(experiment_directory, "latest.pth", decoder, epoch)
        save_optimizer(experiment_directory, "latest.pth", optimizer_all, epoch)
        save_latent_vectors(experiment_directory, "latest.pth", lat_vecs, epoch)

    def signal_handler(sig, frame):
        print("Stopping early...")
        sys.exit(0)

    def adjust_learning_rate(lr_schedules, optimizer, epoch):

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedules[i].get_learning_rate(epoch)

    signal.signal(signal.SIGINT, signal_handler)

    num_samp_per_scene = specs["SamplesPerScene"]
    subsample_random_seed = specs["RandomSubsampleSeed"]
    scene_per_batch = specs["ScenesPerBatch"]
    enforce_minmax = True



    code_bound = get_spec_with_default(specs, "CodeBound", None)
    decoder = DeepSDF(latent_size, **specs["NetworkSpecs"]).cpu()
    # decoder, _ = load_decoder(experiment_directory, 'latest')
    print("training with {} CPU(s)".format(torch.cpu.device_count()))
    decoder = torch.nn.DataParallel(decoder)
    # decoder.double()

    num_epochs = specs["NumEpochs"]
    log_frequency = get_spec_with_default(specs, "LogFrequency", 1)

    with open(train_split_file, "r") as f:
        train_split = json.load(f)

    sdf_dataset = lib.data.SDFSamples(data_source, train_split, num_samp_per_scene, random_seed=subsample_random_seed)

    num_data_loader_threads = get_spec_with_default(specs, "DataLoaderThreads", 1)
    print("loading data with {} threads".format(num_data_loader_threads))

    sdf_loader = data_utils.DataLoader(
        sdf_dataset,
        batch_size=scene_per_batch,
        shuffle=True,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    sdf_loader_reconstruction = data_utils.DataLoader(
        sdf_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_data_loader_threads,
        drop_last=False,
    )

    num_scenes = len(sdf_dataset)

    print("There are {} scenes".format(num_scenes))
    print(decoder)

    lat_vecs = torch.nn.Embedding(num_scenes, latent_size).cpu()
    torch.nn.init.normal_(
        lat_vecs.weight.data,
        0.0,
        get_spec_with_default(specs, "CodeInitStdDev", 1.0) / math.sqrt(latent_size),
    )

    loss_l1 = torch.nn.L1Loss(reduction="sum")

    optimizer_all = torch.optim.Adam(
        [
            {
                "params": decoder.parameters(),
                "lr": lr_schedules[0].get_learning_rate(0),
            },
            {
                "params": lat_vecs.parameters(),
                "lr": lr_schedules[1].get_learning_rate(0),
            },
        ]
    )

    loss_log = []

    start_epoch = 1

    print("starting from epoch {}".format(start_epoch))
    print(
        "Number of decoder parameters: {}".format(
            sum(p.data.nelement() for p in decoder.parameters())
        )
    )
    print(
        "Number of shape code parameters: {} (# codes {}, code dim {})".format(
            lat_vecs.num_embeddings * lat_vecs.embedding_dim,
            lat_vecs.num_embeddings,
            lat_vecs.embedding_dim,
        )
    )
    # train parameterization

    test_loss = []

    for epoch in range(start_epoch, num_epochs + 1):

        adjust_learning_rate(lr_schedules, optimizer_all, epoch)

        train(decoder, sdf_loader, optimizer_all, epoch, specs, enforce_minmax, num_samp_per_scene, loss_l1, loss_log,
              lat_vecs)

        # decoder.eval()
        # with torch.no_grad():
        #
        #     for sdf_test_data, test_indices, test_names in sdf_loader_test:
        #         pass

        print("epoch {}...".format(epoch))

        if epoch % log_frequency == 0:
            save_latest(epoch)
            save_logs(
                experiment_directory,
                loss_log,
                epoch,
            )

    # store reconstruction
    # decoder.eval()
    # reconstruction_dir = get_reconstruction_dir(experiment_directory, True)
    # for sdf_data, indices, name in sdf_loader_reconstruction:
    #     latent = lat_vecs(indices.cpu()).squeeze(0)
    #     mesh_filename = get_mesh_filename(reconstruction_dir, name[0])
    #     print("Reconstructing {}...".format(mesh_filename))
    #     with torch.no_grad():
    #         create_mesh(decoder, latent, N=resolution, output_mesh=False, filename=mesh_filename)

    print("Done!")


def train(decoder, sdf_loader, optimizer_all, epoch, specs, enforce_minmax, num_samp_per_scene, loss_l1, loss_log,
          lat_vecs):

    decoder.train()

    grad_clip = get_spec_with_default(specs, "GradientClipNorm", None)
    clamp_dist = specs["ClampingDistance"]
    minT = -clamp_dist
    maxT = clamp_dist
    do_code_regularization = get_spec_with_default(specs, "CodeRegularization", True)
    code_reg_lambda = get_spec_with_default(specs, "CodeRegularizationLambda", 1e-4)
    for sdf_data, indices, names in sdf_loader:
        optimizer_all.zero_grad()

        sdf_data = sdf_data.reshape(-1, 4)
        num_sdf_samples = sdf_data.shape[0]
        sdf_data.requires_grad = False
        xyz = sdf_data[:, 0:3]
        sdf_gt = sdf_data[:, 3].unsqueeze(1)

        if enforce_minmax:
            sdf_gt = torch.clamp(sdf_gt, minT, maxT)

        indices = indices.unsqueeze(-1).repeat(1, num_samp_per_scene).view(-1)
        batch_vecs = lat_vecs(indices.cpu())

        pred_sdf = decoder(batch_vecs, xyz.cpu())

        if enforce_minmax:
            pred_sdf = torch.clamp(pred_sdf, minT, maxT)

        batch_loss = loss_l1(pred_sdf, sdf_gt.cpu()) / num_sdf_samples

        if do_code_regularization:
            l2_size_loss = torch.sum(torch.norm(batch_vecs, dim=1))
            reg_loss = (
                               code_reg_lambda * min(1, epoch / 100) * l2_size_loss
                       ) / num_sdf_samples

            batch_loss = batch_loss + reg_loss.cpu()

        batch_loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

        optimizer_all.step()

        loss_log.append(batch_loss.item())


if __name__ == "__main__":
    import argparse

    arg_parser = argparse.ArgumentParser(description="Train MeshSDF.")
    arg_parser.add_argument(
        "--experiment",
        "-e",
        dest="experiment_directory",
        required=True,
        help="The experiment directory. This directory should include "
             + "experiment specifications in 'specs.json', and logging will be "
             + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resolution",
        dest="resolution",
        default=128,
        help="Marching cubes resolution for reconstructed surfaces.",
    )
    args = arg_parser.parse_args()
    main_function(args.experiment_directory, args.resolution)
