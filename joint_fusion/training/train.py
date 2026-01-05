from joint_fusion.models.loss_functions import JointLoss
from joint_fusion.models.multimodal_network import MultimodalNetwork
from .learning_scheduler import CosineAnnealingWarmRestartsDecay
from joint_fusion.utils.utils import mixed_collate
from .pretraining import (
    create_data_loaders,
    create_stratified_survival_folds,
    extract_survival_data,
)
from .visualizations import plot_survival_distributions
from joint_fusion.config.config_manager import Config, ConfigManager

import gc
import numpy as np
import torch
import os
import wandb
import time
import joblib
from datetime import datetime

from torch import nn
from torch.utils.data import Subset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sksurv.metrics import concordance_index_censored
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.autograd.set_detect_anomaly(True)

current_time = datetime.now().strftime("%y_%m_%d_%H_%M")


def print_gpu_memory_usage():
    """Print detailed GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")


def train_nn(config: Config, h5_file, device, plot_distributions=True):

    from pathlib import Path
    from dotenv import load_dotenv

    env_path: Path = Path.cwd() / ".env"

    load_dotenv(env_path)

    project = os.getenv("WANDB_PROJECT")
    entity = os.getenv("WANDB_ENTITY")

    wandb.init(
        project=project,
        entity=entity,
        config={
            "num_folds": opt.n_folds,
            "num_epochs": opt.num_epochs,
            "batch_size": opt.batch_size,
            "val_batch_size": opt.val_batch_size,
            "lr": opt.lr,
            "mlp_layers": opt.mlp_layers,
            "dropout": opt.dropout,
            "fusion_type": opt.fusion_type,
            "joint_embedding": opt.joint_embedding,
            "embedding_dim_wsi": opt.embedding_dim_wsi,
            "embedding_dim_omic": opt.embedding_dim_omic,
            "input_mode": opt.input_mode,
            "stratified_cv": True,
            "use_pretrained_omic": opt.use_pretrained_omic,
            "omic_checkpoint_path": (
                opt.omic_checkpoint_path if opt.use_pretrained_omic else None
            ),
        },
    )

    if opt.scheduler == "cosine_warmer":

        wandb.config.update(
            {
                "scheduler": opt.scheduler,
                "scheduler_params": {
                    "t_0": opt.t_0,
                    "t_mult": opt.t_mult,
                    "eta_min": opt.eta_min,
                },
            },
        )

    elif opt.scheduler == "exponential":

        wandb.config.update(
            {
                "scheduler": opt.scheduler,
                "scheduler_params": {
                    "exp_gamma": opt.exp_gamma,
                },
            },
        )
    elif opt.scheduler == "step_lr":

        wandb.config.update(
            {
                "scheduler": opt.scheduler,
                "scheduler_params": {"exp_gamma": opt.exp_gamma, "step": opt.lr_step},
            }
        )
    current_time = datetime.now()

    # If user provided timestamp then use for consistency
    if opt.timestamp:

        checkpoint_dir = "checkpoints/checkpoint_" + opt.timestamp
        os.makedirs(checkpoint_dir, exist_ok=True)

    else:

        checkpoint_dir = "checkpoints/checkpoint_" + current_time.strftime(
            "%Y-%m-%d-%H-%M-%S"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, _, test_loader = create_data_loaders(opt, h5_file)
    dataset = (
        train_loader.dataset
    )  # use only the training dataset for CV during training [both the validation and test splits can be used for testing]

    total_samples = len(dataset)
    print(f"total training data size: {total_samples} samples")

    # Extract survival data for stratification
    print("Extracting survival data for startified CV...")
    times, events = extract_survival_data(dataset)

    print(f"Survival data summary:")
    print(f"  - Total samples: {len(times)}")
    print(f"  - Events: {events.sum()} ({events.mean():.1%})")
    print(f"  - Censored: {len(events) - events.sum()} ({1 - events.mean():.1%})")
    print(f"  - Median survival time: {np.median(times):.1f} days")
    print(f"  - Time range: {times.min():.1f} - {times.max():.1f} days")

    try:
        folds = create_stratified_survival_folds(
            times=times,
            events=events,
            n_splits=opt.n_folds,
            n_time_bins=10,
        )
        print(f"Successfully created {len(folds)} stratified folds")

    except Exception as e:
        print(f"Error creating stratified folds: {e}")
        # Fallback to regular k-fold if stratified fails
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=opt.n_folds, shuffle=True, random_state=6)
        folds = list(kf.split(np.arange(len(dataset))))
        print("Falling back to regular K-Fold CV")

    num_folds = opt.n_folds
    num_epochs = opt.num_epochs

    # Obtain Validation CI and Loss avg at each epoch
    ci_average_by_epoch = np.zeros(opt.num_epochs, dtype=np.float32)
    loss_average_by_epoch = np.zeros(opt.num_epochs, dtype=np.float32)

    # need to keep fold index at 0 for the sake of the averaging by epoch
    if opt.use_mixed_precision:
        amp_scaler = GradScaler()
        print("Using mixed precision training with StandardScaler")

    # Step counter for wandb
    step_counter = 0
    # Main fold iteration loop
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{len(folds)}")
        print(f"{'='*60}")

        # Print fold statistics
        print(f"Training samples: {len(train_idx)}")
        print(f"Validation samples: {len(val_idx)}")

        # Extract survival data for this fold to verify stratification
        times_train_fold = times[train_idx]
        events_train_fold = events[train_idx]
        times_val_fold = times[val_idx]
        events_val_fold = events[val_idx]

        train_event_rate = events_train_fold.mean()
        val_event_rate = events_val_fold.mean()

        print(f"Train event rate: {train_event_rate:.3f}")
        print(f"Validation event rate: {val_event_rate:.3f}")
        print(f"Train median time: {np.median(times_train_fold):.1f} days")
        print(f"Validation median time: {np.median(times_val_fold):.1f} days")

        # Plot distributions if requested
        if plot_distributions:
            plot_survival_distributions(
                times_train_fold,
                events_train_fold,
                times_val_fold,
                events_val_fold,
                fold_idx,
                save_dir=os.path.join(checkpoint_dir, "fold_distributions"),
            )

        print_gpu_memory_usage()  # Monitor memory at start of each fold

        # Create subsets for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # Create data loaders for this fold
        train_loader_fold = DataLoader(
            train_subset,
            batch_size=opt.batch_size,
            collate_fn=mixed_collate,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )
        val_loader_fold = DataLoader(
            val_subset,
            batch_size=opt.val_batch_size,
            collate_fn=mixed_collate,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        # Compute number of batches per epoch
        num_batches_per_epoch = len(train_loader_fold)
        print(
            f"Number of batches per epoch for fold {fold_idx + 1}: {num_batches_per_epoch}"
        )

        # Update wandb config
        wandb.config.update(
            {f"num_batches_per_epoch_fold_{fold_idx}": num_batches_per_epoch},
            allow_val_change=True,
        )

        model = MultimodalNetwork(
            embedding_dim_wsi=opt.embedding_dim_wsi,
            embedding_dim_omic=opt.embedding_dim_omic,
            mode=opt.input_mode,
            fusion_type=opt.fusion_type,
            joint_embedding_type=opt.joint_embedding,
            mlp_layers=opt.mlp_layers,
            dropout=opt.dropout,
        )

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            model = nn.DataParallel(model)
            torch.cuda.set_device(0)

        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
        if opt.scheduler == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=opt.exp_gamma
            )
        elif opt.scheduler == "cosine_warmer":
            scheduler = CosineAnnealingWarmRestartsDecay(
                optimizer,
                T_0=opt.t_0,
                T_mult=opt.t_mult,
                eta_min=opt.eta_min,
                decay_factor=opt.decay_factor,
            )

        elif opt.scheduler == "step_lr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=opt.step_lr_step, gamma=opt.step_lr_gamma
            )
        else:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-6
            )

        joint_loss = JointLoss()

        # Initialize and fit the scaler for this fold
        print(f"Fitting scaler for fold {fold_idx + 1}...")
        scaler = StandardScaler()
        for batch_idx, (tcga_id, _, _, _, x_omic) in enumerate(train_loader_fold):
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f"  Fitting batch {batch_idx + 1}/{len(train_loader_fold)}")
            x_omic_np = x_omic.cpu().numpy()
            scaler.partial_fit(x_omic_np)

        # Save the scaler for the current fold
        scaler_path = os.path.join(checkpoint_dir, f"scaler_fold_{fold_idx}.save")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved for fold {fold_idx + 1} at {scaler_path}")

        # EPOCH TRAINING LOOP for this fold
        for epoch in range(opt.num_epochs):
            print(
                f"\n--- Fold {fold_idx + 1}/{len(folds)}, Epoch {epoch + 1}/{opt.num_epochs} ---"
            )
            start_train_time = time.time()
            model.train()

            # # added to reduce memory requirement
            # if isinstance(model, torch.nn.DataParallel):
            #     model.module.gradient_checkpointing_enable()
            # else:
            #     model.gradient_checkpointing_enable()

            loss_epoch = 0

            # Add debug info for first batch
            print(f"Starting epoch {epoch}, about to iterate over training batches...")
            print(f"Training loader has {len(train_loader_fold)} batches")
            # log the learning rate at the start of the epoch
            current_lr = optimizer.param_groups[0]["lr"]

            # model training in batches for the train dataloader for the current fold
            for batch_idx, (
                tcga_id,
                days_to_event,
                event_occurred,
                x_wsi,
                x_omic,
            ) in enumerate(train_loader_fold):
                # x_wsi is a list of tensors (one tensor for each tile)
                print(
                    f"Total training samples in fold: {len(train_loader_fold.dataset)}"
                )
                print(f"Batch size: {opt.batch_size}")
                print(f"Batch index: {batch_idx} out of {num_batches_per_epoch}")

                # print(f"Before scaling - x_omic shape: {x_omic.shape}")
                # print(f"Before scaling - days_to_event shape: {days_to_event.shape}")
                # print(f"Before scaling - event_occurred shape: {event_occurred.shape}")

                # NOTE: Do not apply standard scaler to omic data
                x_wsi = [x.to(device) for x in x_wsi]
                x_omic = x_omic.to(device)

                print(f"After scaling - x_omic shape: {x_omic.shape}")
                days_to_event = days_to_event.to(device)
                # days_to_last_followup = days_to_last_followup.to(device)
                event_occurred = event_occurred.to(device)

                # print(f"Final - days_to_event shape: {days_to_event.shape}")
                # print(f"Final - event_occurred shape: {event_occurred.shape}")
                print("Days to event: ", days_to_event)
                print("event occurred: ", event_occurred)

                optimizer.zero_grad()

                if opt.use_mixed_precision:
                    with autocast():  # should wrap only the forward pass including the loss calculation
                        predictions, wsi_embedding, omic_embedding = model(
                            opt,
                            tcga_id,
                            x_wsi=x_wsi,
                            x_omic=x_omic,  # Now properly scaled
                        )

                        # print(f"Model predictions shape: {predictions.shape}")
                        loss = joint_loss(
                            predictions.squeeze(),
                            days_to_event,
                            event_occurred,
                            wsi_embedding,
                            omic_embedding,
                        )
                        print("\n loss (train, mixed precision): ", loss.data.item())
                        loss_epoch += loss.data.item()

                    # Mixed precision backward pass
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                else:
                    print(" Not using mixed precision")
                    # model for survival outcome (uses Cox PH partial log likelihood as the loss function)
                    # the model output should be considered as beta*X to be used in the Cox loss function

                    start_time = time.time()

                    predictions, wsi_embedding, omic_embedding = model(
                        opt,
                        tcga_id,
                        x_wsi=x_wsi,  # list of tensors (one for each tile)
                        x_omic=x_omic,
                    )
                    # print(f"predictions: {predictions} from train_test.py")
                    step1_time = time.time()
                    loss = joint_loss(
                        predictions.squeeze(),
                        # predictions are not survival outcomes, rather log-risk scores beta*X
                        days_to_event,
                        event_occurred,
                        wsi_embedding,
                        omic_embedding,
                    )  # Cox partial likelihood loss for survival outcome prediction
                    print("\n loss (train): ", loss.data.item())
                    step2_time = time.time()
                    loss_epoch += (
                        loss.data.item()
                    )  # * len(tcga_id)  # multiplying loss by batch size for accurate epoch averaging
                    # backpropagate loss through the entire model arch upto the inputs
                    loss.backward(
                        retain_graph=True if epoch == 0 and batch_idx == 0 else False
                    )  # tensors retained to allow backpropagation for torchhviz (for visualizing the graph)
                    optimizer.step()
                    torch.cuda.empty_cache()

                    step3_time = time.time()
                    print(
                        f"(in train_nn) Step 1: {step1_time - start_time:.4f}s, Step 2: {step2_time - step1_time:.4f}s, Step 3: {step3_time - step2_time:.4f}s"
                    )

            train_loss = loss_epoch / len(
                train_loader_fold.dataset
            )  # average training loss per sample for the epoch

            scheduler.step()  # step scheduler after each epoch
            print("\n train loss over epoch: ", train_loss)
            end_train_time = time.time()
            train_duration = end_train_time - start_train_time

            # return here for profile
            if opt.profile:
                return model, optimizer
            # check validation for all epochs >= 0
            if epoch >= 0:

                # save model once every 5 epochs
                if (epoch + 1) % 5 == 0 and epoch > 0:
                    checkpoint_path = os.path.join(
                        checkpoint_dir, f"checkpoint_fold_{fold_idx}_epoch_{epoch}.pth"
                    )
                    torch.save(
                        {
                            "epoch": epoch,
                            "fold": fold_idx,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        checkpoint_path,
                    )
                    print(
                        f"Checkpoint saved at epoch {epoch}, fold {fold_idx}, to {checkpoint_path}"
                    )

                # before validation to get the dynamic weights, but only if the mode is "wsi_omic"

                start_val_time = time.time()

                # get predictions on the validation dataset
                model.eval()
                val_loss_epoch = 0.0
                all_predictions = []
                all_times = []
                all_events = []

                with torch.no_grad():  # inference on the validation data
                    for batch_idx, (
                        tcga_id,
                        days_to_event,
                        event_occurred,
                        x_wsi,
                        x_omic,
                    ) in enumerate(val_loader_fold):
                        # x_wsi is a list of tensors (one tensor for each tile)
                        print(f"Batch size: {len(val_loader_fold.dataset)}")
                        print(
                            f"Validation Batch index: {batch_idx + 1} out of {np.ceil(len(val_loader_fold.dataset) / opt.val_batch_size)}"
                        )

                        x_wsi = [x.to(device) for x in x_wsi]

                        x_omic = x_omic.to(device)

                        days_to_event = days_to_event.to(device)
                        event_occurred = event_occurred.to(device)
                        print("Days to event: ", days_to_event)
                        print("event occurred: ", event_occurred)
                        outputs, wsi_embedding, omic_embedding = model(
                            opt,
                            tcga_id,
                            x_wsi=x_wsi,  # list of tensors (one for each tile)
                            x_omic=x_omic,
                        )
                        loss = joint_loss(
                            outputs.squeeze(),
                            # predictions are not survival outcomes, rather log-risk scores beta*X
                            days_to_event,
                            event_occurred,
                            wsi_embedding,
                            omic_embedding,
                        )  # Cox partial likelihood loss for survival outcome prediction
                        print("\n loss (validation): ", loss.data.item())
                        val_loss_epoch += loss.data.item() * len(tcga_id)
                        all_predictions.append(outputs.squeeze())
                        all_times.append(days_to_event)
                        all_events.append(event_occurred)

                    val_loss = val_loss_epoch / len(val_loader_fold.dataset)

                    all_predictions = torch.cat(all_predictions)
                    all_times = torch.cat(all_times)
                    all_events = torch.cat(all_events)

                    # convert to numpy arrays for CI calculation
                    all_predictions_np = all_predictions.cpu().numpy()
                    all_times_np = all_times.cpu().numpy()
                    all_events_np = all_events.cpu().numpy()
                    event_rate = all_events_np.mean()

                    c_index = concordance_index_censored(
                        all_events_np.astype(bool), all_times_np, all_predictions_np
                    )
                    print(f"Validation loss: {val_loss}, CI: {c_index[0]}")

                    end_val_time = time.time()
                    val_duration = end_val_time - start_val_time

                    # since fold starts indexing at 0
                    ci_average_by_epoch[epoch] = (
                        (fold_idx) * ci_average_by_epoch[epoch] + c_index[0]
                    ) / (fold_idx + 1)
                    loss_average_by_epoch[epoch] = (
                        fold_idx * loss_average_by_epoch[epoch] + val_loss
                    ) / (fold_idx + 1)

                    mode = (
                        model.module.mode
                        if isinstance(model, nn.DataParallel)
                        else model.mode
                    )
                    if mode == "wsi_omic":
                        wsi_weight_val = (
                            model.module.wsi_weight.item()
                            if isinstance(model.module.wsi_weight, torch.Tensor)
                            else model.module.wsi_weight
                        )
                        omic_weight_val = (
                            model.module.omic_weight.item()
                            if isinstance(model.module.omic_weight, torch.Tensor)
                            else model.module.omic_weight
                        )
                        epoch_metrics = {
                            # Losses
                            "Loss/train_epoch": train_loss,
                            "Loss/val_epoch": val_loss,
                            # Performance metrics
                            "CI/validation": c_index[0],
                            "Event_rate/validation": event_rate,
                            # Performance by epoch
                            "CI/validation/epoch/avg": ci_average_by_epoch[epoch],
                            "Loss/validation/epoch/avg": loss_average_by_epoch[epoch],
                            # Time tracking
                            "Time/train_epoch": train_duration,
                            "Time/val_epoch": val_duration,
                            "Time/total_epoch": train_duration + val_duration,
                            # Learning rate
                            "LR": optimizer.param_groups[0]["lr"],
                            # Metadata
                            "fold": fold_idx,
                            "epoch": epoch,
                            "wsi_weight": wsi_weight_val,
                            "omic_weight": omic_weight_val,
                        }
                    else:
                        epoch_metrics = {
                            # Losses
                            "Loss/train_epoch": train_loss,
                            "Loss/val_epoch": val_loss,
                            # Performance metrics
                            "CI/validation": c_index[0],
                            "Event_rate/validation": event_rate,
                            # Performance by epoch
                            "CI/validation/epoch/avg": ci_average_by_epoch[epoch],
                            "Loss/validation/epoch/avg": loss_average_by_epoch[epoch],
                            # Time tracking
                            "Time/train_epoch": train_duration,
                            "Time/val_epoch": val_duration,
                            "Time/total_epoch": train_duration + val_duration,
                            # Learning rate
                            "LR": optimizer.param_groups[0]["lr"],
                            # Metadata
                            "fold": fold_idx,
                            "epoch": epoch,
                        }
                    # Log all metrics
                    wandb.log(epoch_metrics, step=step_counter)

                    # increment step counter
                    step_counter += 1

        torch.cuda.empty_cache()
        gc.collect()
        print(
            f"Memory cleared after fold {fold_idx} - model will be recreated for next fold"
        )

    wandb.finish()
    return model, optimizer
