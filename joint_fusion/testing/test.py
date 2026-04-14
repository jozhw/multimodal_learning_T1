import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import logging

from pathlib import Path
from sksurv.metrics import concordance_index_censored
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from joint_fusion.utils.utils import parse_tile_coordinates

logger = logging.getLogger(__name__)


def setup_output_dirs(base_path):
    output_dirs = {
        "saliency": str(base_path / "saliency_maps_6sep"),
        "ig": str(base_path / "IG_6sep"),
        "attention": str(base_path / "attention_scores"),
        "attention_tile_scores": str(base_path / "attention_tile_scores"),
        "saliency_tile_scores": str(base_path / "saliency_tile_scores"),
    }
    for output_dir in output_dirs.values():
        os.makedirs(output_dir, exist_ok=True)
    return output_dirs


def unpack_test_batch(batch):
    if len(batch) == 6:
        tcga_id, days_to_event, event_occurred, x_wsi, x_omic, tile_names = batch
        tile_names = list(tile_names[0])
    else:
        tcga_id, days_to_event, event_occurred, x_wsi, x_omic = batch
        tile_names = None

    return tcga_id, days_to_event, event_occurred, x_wsi, x_omic, tile_names


def move_batch_to_device(x_wsi, x_omic, days_to_event, event_occurred, device):
    return (
        x_wsi.to(device),
        x_omic.to(device),
        days_to_event.to(device),
        event_occurred.to(device),
    )


def validate_attribution_config(config):
    needs_single_item_batches = any(
        (
            config.testing.calc_saliency_maps,
            config.testing.calc_IG,
            getattr(config.testing, "calc_attention_maps", False),
        )
    )
    if needs_single_item_batches and config.testing.test_batch_size != 1:
        raise ValueError(
            "Attribution export currently requires testing.test_batch_size == 1"
        )


def denormalize_image(image, mean, std):
    mean = np.array(mean)
    std = np.array(std)
    image = (image * std) + mean
    image = np.clip(image, 0, 1)
    return image


def calc_integrated_gradients(
    model, config, tcga_id, x_omic, x_wsi, baseline=None, steps=10
):
    # baseline = None
    # baseline.shape
    # torch.Size([1, 19962])
    # set_trace()
    if baseline is None:
        logger.info("** using trivial zero baseline for IG **")
        baseline = torch.zeros_like(x_omic).to(x_omic.device)
    else:
        logger.info(
            "** Using mean gene expression values over training samples as the baseline for IG **"
        )
        baseline = torch.from_numpy(baseline).to(x_omic.device)
        baseline = baseline.float()
        baseline = baseline * torch.ones_like(x_omic).to(x_omic.device)
    # set_trace()
    logger.info(f"CALCULATING INTEGRATED GRADIENTS OVER {steps} steps")
    scaled_inputs = [
        baseline + (float(i) / steps) * (x_omic - baseline) for i in range(steps + 1)
    ]
    gradients = []
    steps_index = 0
    for scaled_input in scaled_inputs:
        logger.info(f"steps_index: {steps_index}")
        scaled_input = scaled_input.clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            pred, _, _, _ = model(x_wsi=x_wsi, x_omic=scaled_input)
            logger.info(f"prediction: {pred}")
            output = pred.sum()
            output.backward()
        gradients.append(scaled_input.grad.detach().cpu().numpy())
        steps_index += 1
    # set_trace()
    avg_gradients = np.mean(gradients[:-1], axis=0)
    integrated_grads = (
        x_omic.detach().cpu().numpy() - baseline.detach().cpu().numpy()
    ) * avg_gradients

    return integrated_grads


def run_inference(model, x_wsi, x_omic, return_attention=False):
    outputs_tuple = model(
        x_wsi=x_wsi,
        x_omic=x_omic,
        return_attention=return_attention,
    )
    if return_attention:
        outputs, wsi_embedding, omic_embedding, _, attention_weights = outputs_tuple
        attention_scores = attention_weights[0].squeeze().detach().cpu().numpy()
    else:
        outputs, wsi_embedding, omic_embedding, _ = outputs_tuple
        attention_scores = None

    return outputs, wsi_embedding, omic_embedding, attention_scores


def plot_saliency_maps(
    saliency,
    image,
    tcga_id,
    tile_name,
    patch_id,
    output_dir,
    threshold=0.8,
):
    saliency = saliency.detach().cpu().numpy()
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    # denormalize the image based on normalization factors used during transformations of the test images
    mean = [0.70322989, 0.53606487, 0.66096631]
    std = [0.21716536, 0.26081574, 0.20723464]
    image = denormalize_image(image, mean, std)

    # normalize the saliency map to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("original patch")

    # overlay saliency map on the image
    ax = plt.subplot(1, 3, 2)
    img = ax.imshow(image)
    # saliency_overlay = ax.imshow(saliency, cmap="hot", alpha=0.5)
    saliency_overlay = ax.imshow(np.clip(saliency, threshold, 1), cmap="hot", alpha=0.5)
    cbar = plt.colorbar(saliency_overlay, ax=ax)
    cbar.set_label("saliency value", rotation=270, labelpad=15)

    plt.title("saliency map overlay")

    # plot the saliency map alone
    plt.subplot(1, 3, 3)
    plt.imshow(saliency, cmap="hot")
    plt.colorbar(label="saliency value")
    plt.title("saliency map only")

    tile_stem = (
        Path(tile_name).stem if tile_name is not None else f"tile_{patch_id:04d}"
    )
    save_path = os.path.join(
        output_dir, f"saliency_overlay_{tcga_id[0]}_{tile_stem}.png"
    )
    plt.savefig(save_path)
    logger.info(f"saved saliency overlay to {save_path}")

    plt.close()


def interpret_omic():

    # not used since the code is already simple enough

    pass


def interpret_wsi(
    patient_tiles, tcga_id, output_dir_saliency, tile_names=None, max_patches=10
):
    if patient_tiles.grad is None:
        raise RuntimeError("Gradients have not been computed for the WSI tiles.")

    saliency_maps = patient_tiles.grad.detach().abs().amax(dim=1)
    saliency_scores = saliency_maps.mean(dim=(1, 2)).cpu().numpy()

    logger.info("OBTAINING SALIENCY MAPS")
    for patch_idx in range(min(patient_tiles.shape[0], max_patches)):
        logger.info(
            f"Generating saliency map for patch index {patch_idx} out of {max_patches}"
        )
        tile_name = tile_names[patch_idx] if tile_names is not None else None
        plot_saliency_maps(
            saliency_maps[patch_idx],
            patient_tiles[patch_idx],
            tcga_id,
            tile_name,
            patch_idx,
            output_dir_saliency,
        )

    return saliency_scores


def build_tile_metadata(tile_names):
    if tile_names is None:
        return None

    xs = []
    ys = []
    for tile_name in tile_names:
        x_coord, y_coord = parse_tile_coordinates(tile_name)
        xs.append(x_coord)
        ys.append(y_coord)

    return {
        "tile_names": np.asarray(tile_names, dtype=object),
        "x_coords": np.asarray(xs, dtype=np.int32),
        "y_coords": np.asarray(ys, dtype=np.int32),
    }


def save_tile_scores(tcga_id, tile_names, output_dir, score_name, scores):
    payload = build_tile_metadata(tile_names)
    if payload is None or scores is None:
        return

    payload[score_name] = np.asarray(scores, dtype=np.float32)

    save_path = os.path.join(output_dir, f"{score_name}_{tcga_id[0]}.npz")
    np.savez_compressed(save_path, **payload)
    logger.info(f"Saved tile-level attribution metadata to {save_path}")


def export_attention_maps(tcga_id, attention_scores, output_dir):
    if attention_scores is None:
        return

    save_path = os.path.join(output_dir, f"attention_scores_{tcga_id[0]}.npy")
    np.save(save_path, attention_scores)
    logger.info(f"Saved attention scores for {tcga_id[0]} to {save_path}")


def export_saliency_maps(model, device, tcga_id, x_wsi, x_omic, output_dir, tile_names):
    x_wsi = x_wsi.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    with torch.enable_grad():
        outputs, _, _, _ = run_inference(model, x_wsi, x_omic, return_attention=False)

        if device.type == "cuda":
            allocated_memory = torch.cuda.memory_allocated(device) / (1024**3)
            reserved_memory = torch.cuda.memory_reserved(device) / (1024**3)

            logger.info(f"Allocated memory: {allocated_memory:.2f} GB")
            logger.info(f"Reserved memory: {reserved_memory:.2f} GB")
            logger.info(
                f"Free memory: {torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)} bytes"
            )
            torch.cuda.empty_cache()

        outputs.sum().backward()
        saliency_scores = interpret_wsi(
            x_wsi[0],
            tcga_id,
            output_dir,
            tile_names=tile_names,
        )

    return saliency_scores


def export_integrated_gradients(
    model, config, tcga_id, x_omic, x_wsi, baseline, output_dir
):
    integrated_grads = calc_integrated_gradients(
        model, config, tcga_id, x_omic, x_wsi, baseline=baseline, steps=10
    )
    save_path = os.path.join(output_dir, f"integrated_grads_{tcga_id[0]}.npy")
    np.save(save_path, integrated_grads)
    logger.info(f"Saved integrated gradients for {tcga_id[0]} to {save_path}")


def export_attention_tile_scores(tcga_id, tile_names, output_dir, attention_scores):
    save_tile_scores(
        tcga_id,
        tile_names,
        output_dir,
        "attention_scores",
        attention_scores,
    )


def export_saliency_tile_scores(tcga_id, tile_names, output_dir, saliency_scores):
    save_tile_scores(
        tcga_id,
        tile_names,
        output_dir,
        "saliency_scores",
        saliency_scores,
    )


def test_and_interpret(config, model, test_loader, device, baseline=None):
    model.eval()
    all_tcga_ids = []
    all_predictions = []
    all_times = []
    all_events = []

    base_path = Path(config.testing.output_base_dir)
    output_dirs = setup_output_dirs(base_path)

    excluded_ids = [
        "TCGA-05-4395",
        "TCGA-86-8281",
    ]
    need_saliency = config.testing.calc_saliency_maps
    need_ig = config.testing.calc_IG
    need_attention = config.testing.calc_attention_maps
    validate_attribution_config(config)

    for batch_idx, batch in enumerate(test_loader):
        tcga_id, days_to_event, event_occurred, x_wsi, x_omic, tile_names = (
            unpack_test_batch(batch)
        )
        if tcga_id[0] in excluded_ids:
            logger.info(f"Skipping TCGA ID: {tcga_id}")
            continue

        x_wsi, x_omic, days_to_event, event_occurred = move_batch_to_device(
            x_wsi, x_omic, days_to_event, event_occurred, device
        )

        attention_scores = None
        saliency_scores = None

        with torch.no_grad():
            outputs, _, _, attention_scores = run_inference(
                model,
                x_wsi,
                x_omic,
                return_attention=need_attention,
            )

        if need_attention:
            export_attention_maps(tcga_id, attention_scores, output_dirs["attention"])
            export_attention_tile_scores(
                tcga_id,
                tile_names,
                output_dirs["attention_tile_scores"],
                attention_scores,
            )

        if need_saliency:
            saliency_scores = export_saliency_maps(
                model,
                device,
                tcga_id,
                x_wsi.detach().clone(),
                x_omic,
                output_dirs["saliency"],
                tile_names,
            )

            export_saliency_tile_scores(
                tcga_id,
                tile_names,
                output_dirs["saliency_tile_scores"],
                saliency_scores,
            )

        if need_ig:
            export_integrated_gradients(
                model,
                config,
                tcga_id,
                x_omic,
                x_wsi,
                baseline,
                output_dirs["ig"],
            )

        all_predictions.append(outputs.squeeze().detach().cpu().numpy())

        del outputs
        torch.cuda.empty_cache()
        all_tcga_ids.append(tcga_id)
        all_times.append(days_to_event)
        all_events.append(event_occurred)
        model.zero_grad()
        torch.cuda.empty_cache()

    all_predictions_np = [pred.item() for pred in all_predictions]
    all_events_np = torch.stack(all_events).cpu().numpy()
    all_events_bool_np = all_events_np.astype(bool)
    all_times_np = torch.stack(all_times).cpu().numpy()

    c_index = concordance_index_censored(
        all_events_bool_np.ravel(), all_times_np.ravel(), all_predictions_np
    )

    logger.info(f"CI: {c_index[0]}")

    # set_trace()
    # stratify based on the median risk scores
    median_prediction = np.median(all_predictions_np)
    high_risk_idx = all_predictions_np >= median_prediction
    low_risk_idx = all_predictions_np < median_prediction

    # separate the times and events into high and low-risk groups
    high_risk_times = all_times_np[high_risk_idx]
    high_risk_events = all_events_np[high_risk_idx]
    low_risk_times = all_times_np[low_risk_idx]
    low_risk_events = all_events_np[low_risk_idx]

    # initialize the Kaplan-Meier fitter
    kmf_high_risk = KaplanMeierFitter()
    kmf_low_risk = KaplanMeierFitter()

    # fit
    kmf_high_risk.fit(
        high_risk_times, event_observed=high_risk_events, label="High Risk"
    )
    kmf_low_risk.fit(low_risk_times, event_observed=low_risk_events, label="Low Risk")

    # perform the log-rank test
    log_rank_results = logrank_test(
        high_risk_times,
        low_risk_times,
        event_observed_A=high_risk_events,
        event_observed_B=low_risk_events,
    )

    p_value = log_rank_results.p_value
    logger.info(f"Log-Rank Test p-value: {p_value}")
    logger.info(f"Log-Rank Test statistic: {log_rank_results.test_statistic}")

    plt.figure(figsize=(10, 6))
    kmf_high_risk.plot(ci_show=True, color="blue")
    kmf_low_risk.plot(ci_show=True, color="red")
    plt.title(
        "Patient stratification: high risk vs low risk groups based on predicted risk scores\nLog-rank test p-value: {:.4f}".format(
            p_value
        )
    )
    plt.xlabel("Time (days)")
    plt.ylabel("Survival probability")
    plt.legend()
    output_path = str(Path(config.testing.output_base_dir) / "km_plot_joint_fusion.png")
    plt.savefig(output_path, format="png", dpi=300)

    return None
