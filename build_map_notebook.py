from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def _normalize_cell_source(source: object) -> str:
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def _load_existing_param_cells(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    try:
        nb = nbf.read(path, as_version=4)
    except Exception:
        return {}

    keep: dict[str, str] = {}
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        src = _normalize_cell_source(cell.get("source", ""))
        stripped = src.lstrip()
        if stripped.startswith("# ---- Shared configuration"):
            keep["shared_config"] = src
        elif stripped.startswith("# Radiomap render settings"):
            keep["radiomap_settings"] = src
    return keep


def main() -> None:
    nb = nbf.v4.new_notebook()
    out_path = Path("map.ipynb")
    existing_param_cells = _load_existing_param_cells(out_path)
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            dedent(
                """
                # Map Snapshot From `Nulling_CDF`

                这个 notebook 会复用 `Nulling_CDF.ipynb` 的参数设置，
                自动重抽单次 Monte Carlo snapshot，直到每个 BS sector 都至少有一个 TN pair，
                然后输出：

                1. 初始 drop 平面图
                2. `MUSIC` 检测到的 NTN 和 round-0 paired TN 的平面图
                3. 基于 `layout1.png` 的 detected NTN + paired TN 平面图
                4. 三张 radiomap
                   - no nulling
                   - MUSIC estimated `(u, g)` nulling
                   - true `(u, g)` nulling

                默认绑定 `sionna_gpu` 内核，渲染参数设成了一个可直接执行的中等配置；需要更精细图时再往上调。
                """
            ).strip()
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                import importlib
                import os
                from pathlib import Path

                os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

                import matplotlib.pyplot as plt
                import mitsuba as mi
                import numpy as np
                from matplotlib.colors import ListedColormap
                from matplotlib.lines import Line2D
                from sionna.rt import Camera, RadioMapSolver, Receiver, load_scene

                try:
                    from sionna.rt import transform_mesh
                except Exception:
                    transform_mesh = None

                import SceneConfigSionna
                import nulling_cdf_utils as ncu

                importlib.reload(SceneConfigSionna)
                importlib.reload(ncu)

                from SceneConfigSionna import SceneConfigSionna
                from ntn_music_detection import collapse_cir_to_narrowband, run_music_standard_pipeline

                plt.rcParams.update(
                    {
                        "figure.figsize": (8.0, 6.0),
                        "axes.grid": False,
                        "font.size": 10,
                    }
                )
                """
            ).strip()
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                # ---- Shared configuration (mirrors Nulling_CDF.ipynb, but runs one snapshot only) ----
                scene_path = Path("Denver_scene/10kmwithfigure/10km.xml")
                result_dir = Path("result") / "map_snapshot"
                result_dir.mkdir(parents=True, exist_ok=True)

                # Set to integers if you want reproducible drops.
                # Keep as None if you want fresh TN/NTN/satellite positions every run.
                layout_rng_seed_base = None
                satellite_rng_seed_base = None
                max_drop_attempts = 40
                layout_image_path = Path("Denver_scene/10kmwithfigure/layout1.png")

                # Geometry and deployment
                ntn_rx = 100
                tn_rx = 300
                bs_row = 2
                bs_col = 2
                nbs = bs_row * bs_col
                nsect = 3

                satellite_azimuth_range_deg = (0.0, 360.0)
                satellite_elevation_range_deg = (45.0, 90.0)
                # Carrier and array configuration
                fc = 9.99e9
                tx_antenna_rows = 8
                tx_antenna_cols = 8
                tn_rx_antenna_rows = 1
                tn_rx_antenna_cols = 1
                tx_antennas = tx_antenna_rows * tx_antenna_cols

                # TX sector orientation
                tx_sector_yaw_offset_deg = 0.0
                tx_head_down_deg = 3.0
                tx_sector_roll_deg = 0.0

                tx_sector_yaw_offset_rad = np.deg2rad(tx_sector_yaw_offset_deg)
                tx_sector_pitch_rad = -np.deg2rad(tx_head_down_deg)
                tx_sector_roll_rad = np.deg2rad(tx_sector_roll_deg)

                # Noise and thresholds
                EkT = -174
                B = 100e6
                Tx_power_dbm = 35
                Tx_power = 10 ** ((Tx_power_dbm - 30) / 10)
                Tx_power_handheld_dbm = 23
                Tx_power_handheld = 10 ** ((Tx_power_handheld_dbm - 30) / 10)

                NF = 7
                NF_vsat = 3
                NF_bs = 2
                N0_dBm = EkT + 10 * np.log10(B) + NF
                N0 = 10 ** ((N0_dBm - 30) / 10)
                N0_vsat = 10 ** ((EkT + 10 * np.log10(B) + NF_vsat - 30) / 10)
                N0_bs = 10 ** ((EkT + 10 * np.log10(B) + NF_bs - 30) / 10)

                snr_threshold = -6
                inr_threshold = -6
                h_ntn_th = np.sqrt(10 ** (inr_threshold / 10) * N0_bs * tx_antennas / Tx_power)
                h_tn_th = np.sqrt(10 ** (snr_threshold / 10) * N0_bs * tx_antennas / Tx_power)

                # MUSIC configuration
                music_threshold = 3
                music_covariance_mode = "sample"
                music_num_snapshots = 200
                music_noise_var = N0_bs / Tx_power
                music_rng_seed = 7
                music_source_estimation = "mdl"
                music_energy_ratio = 0.95
                music_reduce_ntn_ant = "max"
                music_user_powers = None
                music_use_sector_orientation = True
                music_sector_pitch_rad = float(tx_sector_pitch_rad)
                music_sector_roll_rad = float(tx_sector_roll_rad)
                music_rotation_order = "zyx"
                music_std_channel_mode = "conj"
                music_std_manifold_label = "yz:+1"
                music_std_flatten_order = "F"
                music_std_scan_mode = "complex"
                music_std_phi_offset_deg = 0.0
                music_std_phi_mirror_about_sector = False
                music_std_horizontal_sign = -1
                music_sector_forward_only = True
                music_sector_forward_cos_min = 0.0
                music_phi_grid_deg = np.arange(0.0, 360.0, 1.0)
                music_theta_grid_deg = np.arange(0.0, 181.0, 1.0)

                # Render one lambda for the map triplet
                lambda_render = float(8e10)
                max_detected_b_terms = "all"

                compute_positions_kwargs = dict(
                    ntn_rx=ntn_rx,
                    tn_rx=tn_rx,
                    centerBS=False,
                    bs_grid=(bs_row, bs_col),
                    bs_boundary=2500,
                    tn_building_ratio=0.6,
                    tn_distance=400,
                    ntn_building_ratio=0.8,
                    plot_grid=False,
                    plot_bs=False,
                    plot_tn=False,
                    plot_ntn=False,
                )

                compute_paths_kwargs = dict(
                    nsect=nsect,
                    fc=fc,
                    tx_rows=tx_antenna_rows,
                    tx_cols=tx_antenna_cols,
                    tn_rx_rows=tn_rx_antenna_rows,
                    tn_rx_cols=tn_rx_antenna_cols,
                    max_depth=0,
                    bandwidth=B,
                    tx_power_dbm=Tx_power_dbm,
                    sector_yaw_offset_rad=tx_sector_yaw_offset_rad,
                    sector_pitch_rad=tx_sector_pitch_rad,
                    sector_roll_rad=tx_sector_roll_rad,
                )

                music_kwargs = dict(
                    tx_rows=int(tx_antenna_rows),
                    tx_cols=int(tx_antenna_cols),
                    nsect=int(nsect),
                    pair_keys=None,
                    detect_num_sources=None,
                    detect_threshold=music_threshold,
                    detect_user_powers=music_user_powers,
                    detect_noise_var=music_noise_var,
                    detect_covariance_mode=music_covariance_mode,
                    detect_num_snapshots=music_num_snapshots,
                    detect_rng_seed=music_rng_seed,
                    detect_source_estimation=music_source_estimation,
                    detect_energy_ratio=music_energy_ratio,
                    detect_reduce_rx_ant=music_reduce_ntn_ant,
                    channel_mode=music_std_channel_mode,
                    manifold_label=music_std_manifold_label,
                    flatten_order=music_std_flatten_order,
                    scan_mode=music_std_scan_mode,
                    phi_offset_deg=music_std_phi_offset_deg,
                    phi_mirror_about_sector=music_std_phi_mirror_about_sector,
                    steering_horizontal_sign=music_std_horizontal_sign,
                    use_sector_orientation=music_use_sector_orientation,
                    sector_pitch_rad=music_sector_pitch_rad,
                    sector_roll_rad=music_sector_roll_rad,
                    rotation_order=music_rotation_order,
                    sector_forward_only=music_sector_forward_only,
                    sector_forward_cos_min=music_sector_forward_cos_min,
                    phi_grid_deg=music_phi_grid_deg,
                    theta_grid_deg=music_theta_grid_deg,
                )

                print(f"scene_path = {scene_path}")
                print(f"layout seed base (user) = {layout_rng_seed_base}")
                print(f"satellite seed base (user) = {satellite_rng_seed_base}")
                print(f"max_drop_attempts = {max_drop_attempts}")
                print(
                    "satellite search range = "
                    f"az {satellite_azimuth_range_deg}, el {satellite_elevation_range_deg}"
                )
                print(f"lambda_render = {lambda_render:.2e}")
                """
            ).strip()
        )
    )
    if "shared_config" in existing_param_cells:
        cells[-1]["source"] = existing_param_cells["shared_config"]

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                scene = load_scene(str(scene_path))
                SceneConfig = SceneConfigSionna(scene)
                SceneConfig.build_coverage_map(grid_size=10, show_xy=True, plot=False)

                def _resolve_base_seed(user_seed):
                    if user_seed is None:
                        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
                    return int(user_seed)

                layout_seed_base_resolved = _resolve_base_seed(layout_rng_seed_base)
                satellite_seed_base_resolved = _resolve_base_seed(satellite_rng_seed_base)
                print(f"layout seed base (resolved) = {layout_seed_base_resolved}")
                print(f"satellite seed base (resolved) = {satellite_seed_base_resolved}")

                selected_attempt_idx = None
                selected_layout_seed = None
                selected_satellite_seed = None

                for attempt_idx in range(int(max_drop_attempts)):
                    layout_seed = int(layout_seed_base_resolved) + int(attempt_idx)
                    satellite_seed = int(satellite_seed_base_resolved) + int(attempt_idx)
                    np.random.seed(layout_seed)
                    sat_rng = np.random.default_rng(satellite_seed)
                    azimuth = float(sat_rng.uniform(*satellite_azimuth_range_deg))
                    elevation = float(sat_rng.uniform(*satellite_elevation_range_deg))

                    compute_positions_kwargs_try = dict(compute_positions_kwargs)
                    compute_positions_kwargs_try["azimuth"] = azimuth
                    compute_positions_kwargs_try["elevation"] = elevation

                    SceneConfig.compute_positions(**compute_positions_kwargs_try)
                    SceneConfig.compute_paths(**compute_paths_kwargs)

                    h_tn_all = collapse_cir_to_narrowband(SceneConfig.a_tn)
                    h_ntn_all = collapse_cir_to_narrowband(SceneConfig.a_ntn)

                    pairing = ncu.pair_tn_to_strongest_tx(
                        h_tn_all,
                        h_tn_th=h_tn_th,
                        tx_antennas=tx_antennas,
                        tx_power=Tx_power,
                        snr_noise_power=N0,
                    )

                    print(
                        f"attempt {attempt_idx:02d}: "
                        f"layout_seed={layout_seed}, sat_seed={satellite_seed}, "
                        f"az={azimuth:.2f}, el={elevation:.2f}, "
                        f"min_count={int(pairing['min_count'])}"
                    )
                    if int(pairing["min_count"]) >= 1:
                        selected_attempt_idx = int(attempt_idx)
                        selected_layout_seed = int(layout_seed)
                        selected_satellite_seed = int(satellite_seed)
                        break

                if selected_attempt_idx is None:
                    raise RuntimeError(
                        "Failed to find a snapshot with at least one TN pair in every BS sector "
                        f"after {int(max_drop_attempts)} attempts."
                    )

                ntn_music_out = run_music_standard_pipeline(
                    h_ntn_all,
                    **music_kwargs,
                )
                music_lookup = ncu.build_music_tx_lookup(
                    ntn_music_out,
                    num_ntn_rx=int(h_ntn_all.shape[0]),
                    num_tx_total=int(h_ntn_all.shape[2]),
                    num_tx_ant=int(h_ntn_all.shape[3]),
                )

                round_out = ncu.run_small_round(
                    h_tn_all,
                    h_ntn_all,
                    pairs_by_tx=pairing["pairs_by_tx"],
                    music_lookup=music_lookup,
                    round_idx=0,
                    lambda_ranges=None,
                    lambda_ranges_music_est=[lambda_render],
                    lambda_ranges_music_real=[lambda_render],
                    tx_power=Tx_power,
                    snr_noise_power=N0,
                    inr_noise_power=N0_vsat,
                    max_detected_b_terms=max_detected_b_terms,
                )

                detected_ntn_idx = np.flatnonzero(np.asarray(round_out["detected_mask"], dtype=bool))
                eval_ntn_idx = np.flatnonzero(np.asarray(round_out["eval_mask"], dtype=bool))
                paired_tx_idx = np.array(sorted(round_out["round_pairs"].keys()), dtype=int)
                paired_bs_idx = np.array([int(tx_idx) // int(nsect) for tx_idx in paired_tx_idx], dtype=int)
                paired_tn_idx = np.array(
                    [int(round_out["round_pairs"][int(tx_idx)]["tn_idx"]) for tx_idx in paired_tx_idx],
                    dtype=int,
                )

                detected_ntn_pos = (
                    np.asarray(SceneConfig.rx_ntn_pos, dtype=float)[detected_ntn_idx]
                    if detected_ntn_idx.size > 0
                    else np.empty((0, 3), dtype=float)
                )
                paired_tn_pos = (
                    np.asarray(SceneConfig.tn_pos, dtype=float)[paired_tn_idx]
                    if paired_tn_idx.size > 0
                    else np.empty((0, 3), dtype=float)
                )

                lambda_key = float(lambda_render)
                raw_precoding = ncu.build_precoding_matrix_from_tx_beams(
                    SceneConfig.tx_name_list,
                    round_out["raw_beams"],
                    nsect=nsect,
                    num_tx_ant=tx_antennas,
                )
                music_est_precoding = ncu.build_precoding_matrix_from_tx_beams(
                    SceneConfig.tx_name_list,
                    round_out["est_beams"][lambda_key],
                    nsect=nsect,
                    num_tx_ant=tx_antennas,
                )
                music_true_precoding = ncu.build_precoding_matrix_from_tx_beams(
                    SceneConfig.tx_name_list,
                    round_out["music_real_beams"][lambda_key],
                    nsect=nsect,
                    num_tx_ant=tx_antennas,
                )

                print(f"selected attempt = {selected_attempt_idx}")
                print(f"selected layout seed = {selected_layout_seed}")
                print(f"selected satellite seed = {selected_satellite_seed}")
                print(f"selected satellite az/el = ({azimuth:.2f}, {elevation:.2f}) deg")
                print("min_count across all BS sectors:", int(pairing["min_count"]))
                print("detected NTN count:", detected_ntn_idx.size)
                print("eval NTN count (true u,g subset):", eval_ntn_idx.size)
                print("round-0 paired TN count:", paired_tn_idx.size)
                print("raw SNR mean:", float(np.mean(round_out["raw_snr_db"])))
                print("MUSIC-est SNR mean:", float(np.mean(round_out["est_snr_db"][lambda_key])))
                print("True-u,g SNR mean:", float(np.mean(round_out["music_real_snr_db"][lambda_key])))
                """
            ).strip()
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                def _bs_color_map(scene_config):
                    palette = ["#d62728", "#ff7f0e", "#8c564b", "#9467bd", "#7f7f7f", "#bcbd22"]
                    return {
                        int(bs_idx): palette[int(bs_idx) % len(palette)]
                        for bs_idx in range(int(scene_config.tx_pos.shape[0]))
                    }


                def plot_initial_drop(scene_config):
                    fig, ax = plt.subplots(figsize=(8.2, 6.5))
                    cmap_bg = ListedColormap(["#e7e5df", "#8f6b53"])
                    bs_colors = _bs_color_map(scene_config)

                    ax.imshow(
                        scene_config.point_type,
                        cmap=cmap_bg,
                        interpolation="nearest",
                        extent=scene_config.extent,
                    )

                    for bs_idx in range(int(scene_config.tx_pos.shape[0])):
                        ax.scatter(
                            [scene_config.tx_pos[bs_idx, 0]],
                            [scene_config.tx_pos[bs_idx, 1]],
                            c=[bs_colors[int(bs_idx)]],
                            marker=(3, 0, -30),
                            s=180,
                            linewidths=1.2,
                            label=f"BS {int(bs_idx)}",
                        )
                    ax.scatter(
                        scene_config.tn_pos[:, 0],
                        scene_config.tn_pos[:, 1],
                        c="#2ca02c",
                        marker="x",
                        s=14,
                        alpha=0.9,
                        label="TN",
                    )
                    ax.scatter(
                        scene_config.rx_ntn_pos[:, 0],
                        scene_config.rx_ntn_pos[:, 1],
                        c="#1f77b4",
                        marker="x",
                        s=14,
                        alpha=0.9,
                        label="NTN",
                    )

                    ax.set_title("Initial Drop Layout")
                    ax.set_xlabel("x (m)")
                    ax.set_ylabel("y (m)")
                    ax.legend(loc="upper right", frameon=True)
                    fig.tight_layout()
                    fig.savefig(result_dir / "layout_initial_drop.png", dpi=220, bbox_inches="tight")
                    plt.show()


                def plot_detected_and_paired(scene_config, detected_idx, paired_idx, paired_tx, paired_bs):
                    fig, ax = plt.subplots(figsize=(8.4, 6.7))
                    cmap_bg = ListedColormap(["#e7e5df", "#8f6b53"])
                    bs_colors = _bs_color_map(scene_config)

                    ax.imshow(
                        scene_config.point_type,
                        cmap=cmap_bg,
                        interpolation="nearest",
                        extent=scene_config.extent,
                    )

                    for bs_idx in range(int(scene_config.tx_pos.shape[0])):
                        ax.scatter(
                            [scene_config.tx_pos[bs_idx, 0]],
                            [scene_config.tx_pos[bs_idx, 1]],
                            c=[bs_colors[int(bs_idx)]],
                            marker=(3, 0, -30),
                            s=170,
                            linewidths=1.2,
                            label=f"BS {int(bs_idx)}",
                        )

                    if detected_idx.size > 0:
                        ax.scatter(
                            scene_config.rx_ntn_pos[detected_idx, 0],
                            scene_config.rx_ntn_pos[detected_idx, 1],
                            c="#0057b8",
                            marker="x",
                            s=34,
                            linewidths=1.5,
                            label="Detected NTN",
                        )

                    if paired_idx.size > 0:
                        ax.scatter(
                            scene_config.tn_pos[paired_idx, 0],
                            scene_config.tn_pos[paired_idx, 1],
                            c="#2ca02c",
                            marker="x",
                            s=28,
                            linewidths=1.2,
                            label="Paired TN",
                        )

                    for tn_idx, tx_idx, bs_idx in zip(
                        paired_idx.tolist(),
                        paired_tx.tolist(),
                        paired_bs.tolist(),
                    ):
                        color = bs_colors[int(bs_idx)]
                        x, y = scene_config.tn_pos[int(tn_idx), :2]
                        ax.scatter(
                            [x],
                            [y],
                            facecolors="none",
                            edgecolors=[color],
                            marker="o",
                            s=165,
                            linewidths=2.0,
                        )
                        ax.text(
                            x + 18.0,
                            y + 18.0,
                            f"b{int(bs_idx)}s{int(tx_idx) % int(nsect)}",
                            color=color,
                            fontsize=8,
                            weight="bold",
                        )

                    ax.set_title("Detected NTN And Round-0 Paired TN")
                    ax.set_xlabel("x (m)")
                    ax.set_ylabel("y (m)")
                    ax.legend(loc="upper right", frameon=True)
                    fig.tight_layout()
                    fig.savefig(result_dir / "layout_detected_and_paired.png", dpi=220, bbox_inches="tight")
                    plt.show()


                plot_initial_drop(SceneConfig)
                plot_detected_and_paired(
                    SceneConfig,
                    detected_ntn_idx,
                    paired_tn_idx,
                    paired_tx_idx,
                    paired_bs_idx,
                )
                """
            ).strip()
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                def plot_detected_and_paired_with_layout_image(
                    scene_config,
                    image_path,
                    detected_idx,
                    paired_idx,
                    paired_tx,
                    paired_bs,
                    figure_width_in=3.45,
                    show_bs_labels=False,
                ):
                    image_path = Path(image_path)
                    if not image_path.exists():
                        raise FileNotFoundError(f"Background image not found: {image_path}")

                    bg_img = plt.imread(str(image_path))
                    bs_colors = _bs_color_map(scene_config)

                    fig, ax = plt.subplots(figsize=(figure_width_in, figure_width_in))
                    ax.imshow(bg_img, extent=scene_config.extent, aspect="auto")
                    try:
                        ax.set_box_aspect(1)
                    except Exception:
                        ax.set_aspect("equal", adjustable="box")

                    for bs_idx in range(int(scene_config.tx_pos.shape[0])):
                        x_bs = scene_config.tx_pos[bs_idx, 0]
                        y_bs = scene_config.tx_pos[bs_idx, 1]
                        ax.scatter(
                            [x_bs],
                            [y_bs],
                            c=[bs_colors[int(bs_idx)]],
                            marker=(3, 0, -30),
                            s=135,
                            linewidths=1.0,
                        )
                        if show_bs_labels:
                            label_dx = 0.006 * (scene_config.extent[1] - scene_config.extent[0])
                            label_dy = 0.006 * (scene_config.extent[3] - scene_config.extent[2])
                            ax.text(
                                x_bs + label_dx,
                                y_bs + label_dy,
                                f"{int(bs_idx)}",
                                color=bs_colors[int(bs_idx)],
                                fontsize=7.2,
                                weight="bold",
                                ha="left",
                                va="bottom",
                                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.8),
                            )

                    if detected_idx.size > 0:
                        ax.scatter(
                            scene_config.rx_ntn_pos[detected_idx, 0],
                            scene_config.rx_ntn_pos[detected_idx, 1],
                            c="#0057b8",
                            marker="x",
                            s=24,
                            linewidths=1.2,
                        )

                    if paired_idx.size > 0:
                        ax.scatter(
                            scene_config.tn_pos[paired_idx, 0],
                            scene_config.tn_pos[paired_idx, 1],
                            c="#2ca02c",
                            marker="x",
                            s=20,
                            linewidths=1.1,
                        )

                    for tn_idx, _tx_idx, bs_idx in zip(
                        paired_idx.tolist(),
                        paired_tx.tolist(),
                        paired_bs.tolist(),
                    ):
                        color = bs_colors[int(bs_idx)]
                        x, y = scene_config.tn_pos[int(tn_idx), :2]
                        ax.scatter(
                            [x],
                            [y],
                            facecolors="none",
                            edgecolors=[color],
                            marker="o",
                            s=92,
                            linewidths=1.5,
                        )

                    ax.set_xlabel("x (m)", fontsize=9.2)
                    ax.set_ylabel("y (m)", fontsize=9.2)
                    ax.tick_params(axis="both", which="major", labelsize=8.2)
                    ax.legend(
                        handles=[
                            Line2D(
                                [0],
                                [0],
                                marker="^",
                                linestyle="None",
                                color="0.25",
                                markerfacecolor="0.25",
                                markersize=5.8,
                                label="BS site",
                            ),
                            Line2D(
                                [0],
                                [0],
                                marker="x",
                                linestyle="None",
                                color="#0057b8",
                                markersize=5.8,
                                markeredgewidth=1.2,
                                label="Detected NTN",
                            ),
                            Line2D(
                                [0],
                                [0],
                                marker="x",
                                linestyle="None",
                                color="#2ca02c",
                                markersize=5.4,
                                markeredgewidth=1.1,
                                label="Paired TN",
                            ),
                            Line2D(
                                [0],
                                [0],
                                marker="o",
                                linestyle="None",
                                color="black",
                                markerfacecolor="none",
                                markersize=6.2,
                                markeredgewidth=1.3,
                                label="Serving BS ring",
                            ),
                        ],
                        loc="upper center",
                        ncol=2,
                        fontsize=7.2,
                        frameon=True,
                        framealpha=0.62,
                        facecolor="white",
                        edgecolor="0.75",
                        borderpad=0.3,
                        handletextpad=0.4,
                        columnspacing=0.8,
                        labelspacing=0.35,
                    )
                    fig.tight_layout(pad=0.3)
                    fig.savefig(
                        result_dir / "layout_detected_and_paired_layout1.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.show()


                plot_detected_and_paired_with_layout_image(
                    SceneConfig,
                    layout_image_path,
                    detected_ntn_idx,
                    paired_tn_idx,
                    paired_tx_idx,
                    paired_bs_idx,
                )
                """
            ).strip()
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                def _cast_solver_seed(seed):
                    if seed is None:
                        return None
                    for type_name in ("UInt", "UInt32", "UInt64"):
                        t = getattr(mi, type_name, None)
                        if t is not None:
                            try:
                                return t(int(seed))
                            except Exception:
                                pass
                    try:
                        import drjit as dr

                        return dr.uint32(int(seed))
                    except Exception:
                        return int(seed)


                def _patch_sampler_seed_compat(solver):
                    sampler = getattr(solver, "_sampler", None)
                    if sampler is None:
                        return
                    cls = sampler.__class__
                    if getattr(cls, "_codex_seed_patched", False):
                        return

                    orig_seed = getattr(cls, "seed", None)
                    if orig_seed is None:
                        return

                    def seed_compat(self, seed, wavefront_size=4294967295):
                        try:
                            return orig_seed(self, seed, int(wavefront_size))
                        except TypeError:
                            return orig_seed(self, seed)

                    cls.seed = seed_compat
                    cls._codex_seed_patched = True


                def _get_terrain_mesh(scene_obj, terrain_name="terrain", z_offset=0.1):
                    terrain = None
                    if hasattr(scene_obj, "objects"):
                        if terrain_name in scene_obj.objects:
                            terrain = scene_obj.objects[terrain_name].clone(as_mesh=True)
                        else:
                            for name in scene_obj.objects.keys():
                                if "terrain" in name.lower() or "plane" in name.lower():
                                    terrain = scene_obj.objects[name].clone(as_mesh=True)
                                    break
                    if terrain is None:
                        return None
                    if transform_mesh is not None:
                        try:
                            transform_mesh(terrain, translation=[0, 0, z_offset])
                        except Exception:
                            pass
                    return terrain


                def render_nulling_radiomap(
                    scene_obj,
                    precoding_matrix,
                    *,
                    metric="inr_ntn",
                    cell_size=(20, 20),
                    max_depth=2,
                    samples_per_tx=2**13,
                    use_terrain=True,
                    terrain_name="terrain",
                    terrain_z_offset=2.5,
                    fallback_center=None,
                    fallback_size=None,
                    overlay_tn_pos=None,
                    overlay_ntn_pos=None,
                    tn_color=(0.35, 0.72, 1.0),
                    ntn_color=(0.55, 1.0, 1.0),
                    tn_display_radius=120,
                    ntn_display_radius=100,
                    camera_pos=(0, 0, 5000),
                    camera_look_at=(0, 0, 0),
                    render_resolution=(1800, 1800),
                    render_fov=90,
                    render_num_samples=4,
                    rm_vmin=None,
                    rm_vmax=None,
                    auto_contrast=False,
                    auto_contrast_percentiles=(5.0, 95.0),
                    auto_contrast_eps=1e-30,
                    seed=1,
                    title=None,
                    save_path=None,
                ):
                    for rx_name in list(scene_obj.receivers):
                        scene_obj.remove(rx_name)

                    precoding = np.asarray(precoding_matrix, dtype=np.complex64)
                    if precoding.ndim != 2:
                        raise ValueError(f"precoding_matrix must be 2-D, got {precoding.shape}")

                    nonzero_count = int(np.count_nonzero(np.linalg.norm(precoding, axis=1) > 0))
                    print(f"TX total = {precoding.shape[0]}, nonzero precoders = {nonzero_count}")
                    if nonzero_count == 0:
                        raise RuntimeError("All precoding rows are zero.")

                    rm_solver = RadioMapSolver()
                    _patch_sampler_seed_compat(rm_solver)

                    precoding_vec = (
                        mi.TensorXf(np.ascontiguousarray(precoding.real.astype(np.float32))),
                        mi.TensorXf(np.ascontiguousarray(precoding.imag.astype(np.float32))),
                    )

                    num_tx = int(precoding.shape[0])
                    max_wavefront = (2**32) - 1
                    max_samples_per_tx = max(1, max_wavefront // max(1, num_tx))
                    eff_samples_per_tx = int(min(int(samples_per_tx), int(max_samples_per_tx)))
                    seed_cast = _cast_solver_seed(seed)

                    measurement_surface = None
                    if use_terrain:
                        measurement_surface = _get_terrain_mesh(
                            scene_obj,
                            terrain_name=terrain_name,
                            z_offset=terrain_z_offset,
                        )

                    if measurement_surface is not None:
                        rm = rm_solver(
                            scene_obj,
                            measurement_surface=measurement_surface,
                            max_depth=max_depth,
                            samples_per_tx=eff_samples_per_tx,
                            precoding_vec=precoding_vec,
                            cell_size=cell_size,
                            seed=seed_cast,
                        )
                    else:
                        rm = rm_solver(
                            scene_obj,
                            max_depth=max_depth,
                            samples_per_tx=eff_samples_per_tx,
                            precoding_vec=precoding_vec,
                            cell_size=cell_size,
                            center=list(fallback_center),
                            size=list(fallback_size),
                            orientation=[0, 0, 0],
                            seed=seed_cast,
                        )

                    temp_names = []
                    if overlay_tn_pos is not None and len(overlay_tn_pos) > 0:
                        for i, pos in enumerate(np.asarray(overlay_tn_pos, dtype=float)):
                            name = f"tmp-tn-{i}"
                            scene_obj.add(
                                Receiver(
                                    name=name,
                                    position=pos,
                                    color=tuple(tn_color),
                                    display_radius=tn_display_radius,
                                )
                            )
                            temp_names.append(name)
                    if overlay_ntn_pos is not None and len(overlay_ntn_pos) > 0:
                        for i, pos in enumerate(np.asarray(overlay_ntn_pos, dtype=float)):
                            name = f"tmp-ntn-{i}"
                            scene_obj.add(
                                Receiver(
                                    name=name,
                                    position=pos,
                                    color=tuple(ntn_color),
                                    display_radius=ntn_display_radius,
                                )
                            )
                            temp_names.append(name)

                    if auto_contrast:
                        metric_tensor = getattr(rm, metric)
                        metric_values = np.asarray(metric_tensor, dtype=np.float64)
                        metric_values = np.real(metric_values).reshape(-1)
                        metric_values = metric_values[np.isfinite(metric_values)]
                        metric_values = metric_values[metric_values > float(auto_contrast_eps)]
                        if metric_values.size > 0:
                            metric_values_db = 10.0 * np.log10(
                                np.maximum(metric_values, float(auto_contrast_eps))
                            )
                            p_low = float(auto_contrast_percentiles[0])
                            p_high = float(auto_contrast_percentiles[1])
                            auto_vmin = float(np.percentile(metric_values_db, p_low))
                            auto_vmax = float(np.percentile(metric_values_db, p_high))
                            if auto_vmax <= auto_vmin:
                                auto_vmax = auto_vmin + 1.0
                            rm_vmin = auto_vmin
                            rm_vmax = auto_vmax
                            print(
                                f"auto contrast for {metric}: "
                                f"p{p_low:.1f}={rm_vmin:.2f} dB, p{p_high:.1f}={rm_vmax:.2f} dB"
                            )
                        else:
                            print(f"auto contrast skipped for {metric}: no positive finite samples")

                    cam = Camera(position=list(camera_pos), look_at=list(camera_look_at))
                    scene_obj.render(
                        camera=cam,
                        radio_map=rm,
                        resolution=tuple(render_resolution),
                        fov=float(render_fov),
                        rm_show_color_bar=True,
                        rm_vmin=rm_vmin,
                        rm_vmax=rm_vmax,
                        rm_metric=metric,
                        num_samples=int(render_num_samples),
                    )

                    plt.show()
                    if save_path is not None:
                        plt.gcf().savefig(save_path, dpi=180, bbox_inches="tight")
                        print(f"saved: {save_path}")

                    for name in temp_names:
                        if name in scene_obj.receivers:
                            scene_obj.remove(name)
                    return rm
                """
            ).strip()
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """
                # Radiomap render settings
                # Set the booleans below to choose which radiomap(s) to render.
                run_radiomap_no_nulling = True
                run_radiomap_music_est = True
                run_radiomap_true_ug = True

                radiomap_metric = "inr_ntn"
                radiomap_cell_size = (1, 1)
                radiomap_max_depth = 3
                radiomap_samples_per_tx = 10*10**7
                radiomap_render_resolution = (9000, 9000)
                radiomap_render_num_samples = 16
                radiomap_vmin = -25
                radiomap_vmax = 45
                radiomap_use_auto_contrast = True
                radiomap_auto_contrast_percentiles = (5.0, 95.0)
                radiomap_camera_pos = [0, 0, 5000]
                radiomap_camera_look_at = [0, 0, 0]
                radiomap_fov = 90
                radiomap_terrain_name = "terrain"
                radiomap_terrain_z_offset = 2.5

                fallback_center = [0.0, 0.0, 1.5]
                fallback_size = [
                    float(SceneConfig.extent[1] - SceneConfig.extent[0]),
                    float(SceneConfig.extent[3] - SceneConfig.extent[2]),
                ]

                radiomap_specs = []
                if run_radiomap_no_nulling:
                    radiomap_specs.append(
                        ("No Nulling", raw_precoding, result_dir / "radiomap_no_nulling.png")
                    )
                if run_radiomap_music_est:
                    radiomap_specs.append(
                        (
                            f"MUSIC Estimated (u, g) Nulling, lambda={lambda_key:.1e}",
                            music_est_precoding,
                            result_dir / "radiomap_music_est_nulling.png",
                        )
                    )
                if run_radiomap_true_ug:
                    radiomap_specs.append(
                        (
                            f"True (u, g) Nulling, lambda={lambda_key:.1e}",
                            music_true_precoding,
                            result_dir / "radiomap_true_ug_nulling.png",
                        )
                    )

                print("Radiomaps selected:")
                for title, _precoding, _save_path in radiomap_specs:
                    print(" -", title)
                if len(radiomap_specs) == 0:
                    print(" - none")
                """
            ).strip()
        )
    )
    if "radiomap_settings" in existing_param_cells:
        cells[-1]["source"] = existing_param_cells["radiomap_settings"]

    cells.append(
        nbf.v4.new_code_cell(
            dedent(
                """

                radio_maps = {}
                for title, precoding, save_path in radiomap_specs:
                    print("\\n" + "=" * 80)
                    print(title)
                    radio_maps[title] = render_nulling_radiomap(
                        scene,
                        precoding,
                        metric=radiomap_metric,
                        cell_size=radiomap_cell_size,
                        max_depth=radiomap_max_depth,
                        samples_per_tx=radiomap_samples_per_tx,
                        use_terrain=True,
                        terrain_name=radiomap_terrain_name,
                        terrain_z_offset=radiomap_terrain_z_offset,
                        fallback_center=fallback_center,
                        fallback_size=fallback_size,
                        overlay_tn_pos=paired_tn_pos,
                        overlay_ntn_pos=detected_ntn_pos,
                        camera_pos=radiomap_camera_pos,
                        camera_look_at=radiomap_camera_look_at,
                        render_resolution=radiomap_render_resolution,
                        render_fov=radiomap_fov,
                        render_num_samples=radiomap_render_num_samples,
                        rm_vmin=radiomap_vmin,
                        rm_vmax=radiomap_vmax,
                        auto_contrast=radiomap_use_auto_contrast,
                        auto_contrast_percentiles=radiomap_auto_contrast_percentiles,
                        title=title,
                        save_path=save_path,
                    )
                """
            ).strip()
        )
    )

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "sionna_gpu",
        "language": "python",
        "name": "sionna_gpu",
    }
    nb["metadata"]["language_info"] = {
        "name": "python",
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "pygments_lexer": "ipython3",
        "nbconvert_exporter": "python",
    }

    out_path.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
