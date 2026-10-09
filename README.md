# Planarian smFISH Analysis 3.10 and 4.0

This repository documents the development of a *Schmidtea mediterranea* smFISH analysis workflow from **v3.10 Jupyter notebooks**, operated one image at a time, to **v4.0 Python scripts** for processing multiple images in batches. Big-FISH provides spot detection and dense-region decomposition. Nuclear segmentation and spot assignment are described separately for the StarDist and Cellpose branches.

**Publication provenance:** the manuscript analysis used the StarDist-based workflow originating in v3.10. The Cellpose branch added in v4.0 was benchmarked and gave broadly similar results in the author's comparison, but **Cellpose-derived results were not used for the manuscript analysis**. This is a qualitative account of that comparison, not a claim of numerical identity or a formal equivalence test. A script's inclusion in this repository does not establish that its outputs or statistical tests were used in the paper.

### Development history

| Stage | How it was used | Spot processing | Nuclear analysis |
| --- | --- | --- | --- |
| v3.10 original workflow | Interactive, image-by-image notebooks | Big-FISH, with optional counterstain processing and control-based removal of low-intensity spots | StarDist on nuclear z-projections, followed by label expansion and spot assignment |
| v3.10 simplified workflow | Later routine analysis | Big-FISH detection/decomposition; the additional counterstain/control filtering steps were generally omitted after the author found limited improvement | StarDist-based per-cell analysis retained |
| v4.0 StarDist-label branch | AI-assisted conversion of the earlier workflow into batch-processing scripts | Batch Big-FISH detection/decomposition | Reuses existing StarDist projection labels; supports independent labels or reciprocal linking |
| v4.0 Cellpose branch | Additional implementation and benchmarking | Uses the same spot-array interface | Cellpose-SAM segmentation, z-stitching, artifact cleanup, and calibrated 3D nearest-nucleus assignment; not used for manuscript analysis |

“Big-FISH only” describes the simplified **spot-processing stage**, with the additional counterstain/control filtering disabled. Per-cell quantification still requires nuclear segmentation and spot assignment. Counterstain processing and control-based filtering are distinct optional operations.

The version **3.10** is the workflow version previously referred to as “3.0.” It is separate from the Python version. Workflow versions, individual script `SCRIPT_VERSION` values, and installed package versions are also separate identifiers.

### Where to start

- For the original and simplified notebook workflows, see **Section 1**.
- For batch processing using existing StarDist labels, use the Big-FISH detector in **Section 6, Step 1**, then the projection-label scripts in **Section 9**.
- For the optional Cellpose branch, follow **Sections 5–8**.
- For branch differences and reproducibility notes, see **Section 11**.

## 1. Version 3.10 notebooks

The historical guide is `User Instructions for smFISH analysis 3.10.docx`. It describes environment setup, image preparation, and the original notebook order `0–1–2–3–4–5–6`. Its optional branches and saved parameter cells should be interpreted in light of the later simplified workflow described above.

### Notebook inventory

| File | Purpose | Place in the simplified workflow |
| --- | --- | --- |
| `0_setting_threshold.ipynb` | Scan image folders, estimate Big-FISH thresholds, and inspect threshold distributions; optional control/counterstain scans | Threshold setup or review; disable unused branches |
| `1_Get_spots(1).ipynb` | Detect/decompose smFISH spots; optional control-based intensity filtering, intensity measurements, and diagnostic plots | Run for each image with `controlImage = False` when using the simplified route |
| `2_Nuclei_segmentation(1).ipynb` | Generate nuclear projections and segment them with StarDist2D | Run for each image |
| `3_Counterstaining(1).ipynb` | Detect/decompose spots in the counterstain channel, with optional control filtering | Historical optional branch; skip when counterstain analysis is omitted |
| `4_Counterstain_Assignments(1).ipynb` | Expand nuclear labels; optionally assign counterstain spots and generate counterstain-positive labels | **Still needed to create expanded nuclear labels**, with `counterstainAssignment = False` |
| `5_smFISH_spots_Assignments(1).ipynb` | Assign smFISH spots to nuclear/projection labels and save count arrays | Run after expanded labels exist; optional counterstain/intensity branches |
| `6-Statistics.ipynb` | Historical aggregation, plotting, exports, and statistical exploration | Run only the applicable analysis cells after checking input paths and count filters |
| `7-wntP-2Statistics.ipynb` | wntP-2-specific nuclear/counterstain comparisons | Dataset-specific extension |
| `8_314_Analysis(1).ipynb` | Experiment 314 exports, expression/intensity plots, and comparisons | Dataset-specific extension |
| `9_306_Analysis(1).ipynb` | Experiment 306 exports, expression/intensity plots, and comparisons | Dataset-specific extension |
| `10_322_Analysis(1).ipynb` | Saved notebook named 322; its initial analysis cells point to **332_notum_analysis** | Verify paths and output names before use; retain the historical filename |
| `11_307_Analysis.ipynb` | Experiment 307 aggregation and incision/agarose/amputation comparisons | Dataset-specific extension |

Notebooks 7–11 are analysis extensions, not additional mandatory stages to run in sequence. Some contain reused cells with paths to other experiments. The source snapshots also contain saved errors from prior sessions; saved outputs are not proof of a clean run with the currently displayed settings.

### Environment and input preparation

The notebook workflow uses Big-FISH, StarDist, TensorFlow, CSBDeep, NumPy, SciPy, scikit-image, tifffile, pandas, matplotlib, seaborn, tqdm, and Jupyter/IPython. Some import cells also load Tkinter for file selection. The custom StarDist model is an external dependency: notebook 2 defaults to a model named `stardist_2D` under `modelDirectory`. Model weights are not included in this file set. A `2D_versatile_fluo` alternative exists when `custom_model = False`, but that is a different model choice.

Use the environment and model associated with the original analysis when reproducing it. The Word guide's Apple Silicon/TensorFlow instructions describe the historical setup and are not a pinned environment. In an already configured environment, open the notebooks with:

```bash
jupyter lab
```

Select the appropriate analysis kernel and execute the configured cells in order. The notebooks contain IPython magics and interactive state; they are not command-line Python scripts.

The historical preparation workflow uses ImageJ/Fiji to orient images as needed, select/crop the analysis region and z-range, save ROI information, and split channels into separate TIFF stacks. Preserve matching coordinates across channels and retain the actual preprocessing decisions used for each dataset. Counterstain examples use a `633` channel in addition to nuclear `405` and smFISH `565`.

### Simplified image-by-image run order

1. **Configure threshold estimation in notebook 0 when needed.** Set `experimentDirectory`, filename pattern, and calibration. Set `controlDirectory = None` and `counterstain = False` when those branches are unused. This setup notebook can scan multiple files even though subsequent image analysis was operated manually.
2. **Run notebook 1 for one image.** Set `smFISHChannelPath`, voxel size, detection parameters, and `experimentThreshold`. With `controlImage = False`, retain the Big-FISH decomposed spots without the additional control-intensity rejection step. The supplied notebook has `experimentThreshold = None`, which requests automatic threshold selection; it does not universally use threshold 12.
3. **Run notebook 2 for the same image.** Set the corresponding `smFISHChannelPath`, `nucleiChannelPath`, model location, projection size, and StarDist parameters. Inspect the nuclear labels and example overlays.
4. **Skip notebook 3 when counterstain processing is omitted.** Run notebook 4 with `counterstainAssignment = False` to create `565/results/expanded_labels/`. Update its image and label paths to match the same sample.
5. **Run notebook 5 for that sample.** Set `counterstainAssignment = False`, point to the matching labels, and keep `nuclei_projection_size` consistent. It assigns spots by projection index and saves per-label counts. Check the expansion behavior described below before reproducing a particular run.
6. **Repeat for the remaining images**, then use the applicable aggregation/export/statistics cells. Verify whether the intended output includes all segmented labels or only positive-count labels.

The historical full counterstain route adds notebook 3 and the counterstain-enabled branches of notebooks 4 and 5. Control-based spot filtering is enabled separately through the detection notebooks' control settings. These branches are retained as historical code and require their own paths and runtime checks before reuse.

### Saved settings and outputs

Notebook 2 currently specifies `nuclei_projection_size = 10`, `detection_thresh = (0.35, 0.3)` mapped to StarDist NMS/probability thresholds, and normalization percentiles `(0.5, 99.9)`. Notebook 4 sets `labelExpansionSize = 20` pixels. These are values in the supplied snapshots, not a substitute for experiment-specific run records.

| Output under a sample's `565/results/` | Meaning |
| --- | --- |
| `spots.npy` | Initial detected spots |
| `spots_post_decomposition.npy` | Spot array selected after decomposition and any enabled control filtering |
| `spots_post_decomposition_and_background_removed.npy` | Explicitly named filtered array, written when the control branch is enabled |
| `spot_intensity_array.npy` | Intensity estimates for the selected spots in notebook 1 |
| `nuclei_projection.tif` | Nuclear projection stack used for StarDist segmentation |
| `labels/Nucleus_Labels_*.tif` | StarDist projection labels; IDs are local to each projection |
| `polygons/` and `random_examples/` | ImageJ ROI polygons and segmentation review images |
| `expanded_labels/` | Expanded nuclear labels generated by notebook 4 |
| `assignment_results_of_spots_in_nuclei.npz` | One count array per nuclear projection |
| `concatenated_spot_in_nuclei.npy` | Concatenated count arrays across projections, including zero entries |
| `spot_in_background.npy` | Spots assigned to background in each projection |

The control branch estimates spot intensity from summed raw signal around detected spots and retains decomposed experimental spots whose intensity is at least the mean control-spot intensity. In notebook 1, **both decomposed-spot filenames can contain the filtered array when this branch is enabled**. The filename `spots_post_decomposition.npy` alone therefore does not prove that control filtering was absent.

Many historical statistics cells use `cells[cells > 0]`, whereas the underlying concatenated count arrays retain zero entries. Several notebook comparisons pool per-cell values across images. Confirm the subset and statistical unit used by each cell; the v4.0 sample-mean tests should not be assumed to reproduce every historical notebook statistic.

### Source details relevant to reproduction

The saved notebooks are working analysis snapshots. The following behaviors matter when matching an earlier result:

- **Projection construction:** notebook 2 computes each block end as `start + projection_size - 1` and then iterates with `range(start, end)`. For normal multi-slice blocks this excludes the final slice of the nominal block; it also casts the projected array to `uint8`. The spot-assignment projection index still uses division by the configured projection size. Reconstruct the settings and code actually used before treating a fresh run as identical to archived labels.
- **Label expansion:** notebook 4 expands the labels by 20 pixels, and the supplied notebook 5 loads those expanded labels and calls assignment with `expansion=20` again. The v4.0 projection script expands the original masks once. Check the archived settings when comparing the two routes; selecting `--link-mode none` alone does not establish numerical equivalence.
- **Full-animal export:** the initial NPZ-export cells in notebooks 10 and 11 read only `data[data.files[0]]`. Because notebook 5 writes one NPZ entry per projection, this selects only the first projection when multiple entries exist. For complete counts, use `concatenated_spot_in_nuclei.npy` or explicitly combine all projection arrays. Notebook 10 also includes a later export using the concatenated counts.
- **Re-enabling counterstain assignment:** notebook 5 references `counterstainAssignmentPath` without defining it in the supplied notebook and unpacks two results from a function that now returns three. That optional branch needs reconciliation before it can be run from a clean kernel.

These observations concern the supplied source snapshots; they do not establish which exact cell revisions generated any particular manuscript result. No notebook code has been changed as part of this documentation update.

## 2. Version 4.0 scripts and branches

Version 4.0 was developed by converting the earlier notebooks into scripts with AI assistance, allowing multiple images to be processed in batches. It retains a StarDist-label branch and adds a Cellpose branch. It consists of multiple stage-specific scripts rather than a single pipeline runner.

For the **StarDist-label branch**, run the batch spot detector and reuse or generate StarDist masks with notebook 2, then use `reassign_smfish_old_labels_modes.py`. Its `none` and `reciprocal` modes provide explicit alternatives for treating labels across projections. The supplied v4.0 scripts consume StarDist masks; they do not contain a new batch StarDist inference stage.

For the **Cellpose branch**, run these stages in order:

1. Detect and decompose smFISH spots.
2. Segment the nuclear channel and stitch adjacent z-plane masks.
3. Clean the stitched label stack.
4. Assign spots to cleaned nuclei and generate quantitative results.
5. Collect the per-experiment tables containing all cells, including zeros.

Benchmarking and figure-example export are additional utilities. The Cellpose branch is documented for reuse and benchmarking; its inclusion does not change the publication provenance above.

| Script | Role | Main input → output |
| --- | --- | --- |
| `batch_get_spots_threshold12.py` | Core: spot detection | Raw 565 TIFF → detected/decomposed spot coordinates and QC |
| `cellpose_sam_nuclei_3d(1).py` | Cellpose branch: segmentation and stitching | Raw 405 TIFF → globally labeled nuclear z-planes |
| `clean_cellpose_fragment_artifacts.py` | Cellpose branch: label cleanup | Stitched labels → cleaned labels and removal records |
| `reassign_smfish_3d(1).py` | Cellpose branch: 3D assignment and analysis | Cleaned labels + spots → per-cell counts, per-animal statistics, maps, and QC |
| `export_all_cell_results_including_zero.py` | Cellpose branch: collect all-cell tables | Per-experiment 3D wide tables → five prefixed CSV files |
| `cellpose_threshold_benchmark.py` | Optional: parameter exploration | One nuclear-image ROI → masks, overlays, comparison image, and summary |
| `export_condition_examples_gt20.py` | Optional: full z-stack examples | 3D assignment results + raw channels → one selected crop per experiment/condition |
| `reassign_smfish_old_labels_modes.py` | StarDist-label branch: batch assignment and linking sensitivity | Existing projection labels → counts with or without reciprocal linking |
| `compare_smfish_linking_nonzero_sample_means.py` | StarDist-label branch: compare linking modes | Two projection-analysis roots → sample means, Welch tests, and plots |
| `reassign_smfish_projection_for_now.py` | Earlier temporary projection workflow | Existing projection labels → reciprocal-linked counts and QC |
| `export_all_cells_including_zero.py` | Supplementary: rebuild compatible sample-level exports | Co-located count/QC/completion files → all-cell tables and summaries |

Commands below preserve the supplied filenames, including `(1)`. Keep those names quoted in the shell. If you rename them for your repository, update the commands accordingly.

## 3. Inputs and directory conventions

For each experiment, the expected sample layout is:

| Path relative to the experiment root | Contents |
| --- | --- |
| `Experiment/<condition>/<animal>/405/<animal>_405.tif` | Nuclear channel; the segmentation script expects exactly one TIFF with `405` in its filename in this folder |
| `Experiment/<condition>/<animal>/565/<animal>_565.tif` | smFISH channel |
| `Experiment/<condition>/<animal>/565/results/` | Spot arrays and segmentation output folders |
| `Experiment/<condition>/<animal>/565/results/labels/` | StarDist projection labels for the notebook/projection workflows |

Example condition and animal names are `6h_Incision`, `12h_Amputation`, and `Image1`. The 3D analysis extracts time and injury from condition names. Names containing `held`/`hold_open`/`holdopen` or both `dorsal` and `window` also have supported injury categories. Check the exported metadata when using other naming conventions.

The segmentation and 3D assignment scripts discover samples under `Experiment/<condition>/<animal>` and ignore sibling `Control` trees. A `565/results` directory must already exist for sample discovery.

### Configured experiment roots

All paths below are relative to `/Volumes/Backup Plus/Experiment_results/`.

| Experiment | Gene | Root suffix | Voxel size: X, Y, Z (nm), used for cleanup/3D assignment |
| --- | --- | --- | --- |
| 304 | wnt1 | `304_Analysis_results` | 64, 64, 361 |
| 306 | wnt1 | `306_analysis_results_alter` | 64, 64, 361 |
| 307 | wnt1 | `307_analysis_resutls/307_quantificatoin` | 64, 64, 361 |
| 314 | notum | `314_analysis_results` | 74, 74, 500 |
| 332 | notum | `332_notum_analysis` | 74, 74, 500 |

The spellings `resutls` and `quantificatoin` are the paths present in the source. Preserve them when using the original data drive.

### Coordinates and spot-file selection

- Raw inputs should be single-channel **ZYX** stacks with matching spatial coordinates. Channel registration is not performed by these scripts.
- Spot coordinates are **zero-based `(z, y, x)`** voxel/pixel coordinates. The normal input is an `N × 3` NumPy array. Assignment readers extract the first and last two columns when more columns are present.
- Cellpose label files are `Nucleus_Labels_0000.tif`, `Nucleus_Labels_0001.tif`, etc. Their indices correspond to raw z-planes. Background is `0`; positive values identify nuclei.
- Assignment scripts first look for `spots_post_decomposition_and_background_removed.npy`, then fall back to `spots_post_decomposition.npy`.
- The v4.0 batch detector creates `spots_post_decomposition.npy` without the optional control-intensity filtering. The v3.10 detection notebooks can create background-removed arrays, and notebook 2 creates the StarDist projection labels.
- `spots_post_clustering.npy` includes cluster metadata and is not an interchangeable assignment input.

**Check spot-file provenance before rerunning:** an older background-removed array takes precedence over a newly generated decomposed array. Confirm the selected file in each sample's assignment QC.

The detector can handle several raw TIFFs in one `565` folder by writing `results/<image_stem>/`. Downstream scripts expect the sample's spot array directly in `565/results/`. For the documented end-to-end layout, use one smFISH stack per sample/channel folder, or explicitly reconcile the outputs before assignment.

## 4. Version 4.0 environment

The source uses Python 3.10+ syntax/features. The segmentation script declares **`cellpose>=4.2`** and defaults to model **`cpsam_v2`**. Reuse the working analysis environment when available.

Dependencies inferred from the supplied scripts:

| Stage | Packages |
| --- | --- |
| Spot detection | `big-fish`, `numpy`, `tifffile`, `matplotlib`, `seaborn` |
| Segmentation/benchmark | `cellpose`, `torch`, `numpy`, `tifffile`; `matplotlib` for the benchmark |
| Cleanup/assignment/statistics | `numpy`, `scipy`, `pandas`, `tifffile`, `matplotlib` |
| Full z-stack example export | `numpy`, `pandas`, `tifffile`, `colorcet` |
| Projection workflows | `numpy`, `pandas`, `scikit-image`, `scipy`, `tifffile`; Pillow is an optional TIFF fallback |

Example installation in a dedicated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'cellpose>=4.2' torch big-fish numpy scipy pandas \
  tifffile matplotlib seaborn scikit-image colorcet pillow
```

This is a dependency list, not a validated version lock. Record the actual package versions and model used for analysis. Model weights must be available to the Cellpose installation.

Segmentation supports `--device auto`, `mps`, `cuda`, or `cpu`. The examples use Apple Silicon's `mps` setting. `caffeinate -ims` may be prepended to long commands on macOS to prevent sleep. The downstream assignment calculations do not expose a GPU device option.

Segmentation streams z-planes and uses disk-backed arrays; compressed nuclear TIFFs may require temporary decompressed storage. Big-FISH loads the raw stack and additional arrays into memory, so the workflow is not uniformly streaming.

## 5. Version 4.0 calibration and default parameters

### Spot detection: confirm calibration before use

| Setting in `batch_get_spots_threshold12.py` | Supplied value |
| --- | --- |
| Detection threshold | `12.0` |
| `MINIMUM_DISTANCE`, ZYX pixels | `(2, 2, 2)` |
| `LOG_KERNEL_SIZE`, ZYX pixels | `(1, 1.5, 1.5)` |
| `VOXEL_SIZE`, ZYX nm | **`(361, 75, 75)`** |
| `SPOT_RADIUS`, ZYX nm | `(600, 300, 300)` |
| Dense-region decomposition: alpha, beta, gamma | `0.7, 1, 5` |
| Cluster radius / minimum spots | `250 nm` / `4` |

**The detector's hard-coded voxel size differs from the cleanup/assignment table above.** Confirm the acquisition metadata before regenerating spots. There is no voxel-size CLI option in this detector; `--experiment-dir` changes the scan location but does not update calibration. If the intended metadata are 64/64/361 nm, the detector's ZYX tuple would be `(361, 64, 64)`; for 74/74/500 nm it would be `(500, 74, 74)`. Make any source change deliberately and retain a record of the settings actually used.

### Cellpose branch defaults

| Option | Default |
| --- | --- |
| `--model` | `cpsam_v2` |
| `--diameter` | `90` pixels |
| `--niter` | `500` |
| `--flow-threshold` | `0.8` |
| `--cellprob-threshold` | `-1.75` |
| `--smooth-radius` | `4` |
| `--tile-overlap` | `0.5` |
| `--normalization` | `plane` |
| `--stitch-threshold` | `0.25` |
| `--batch-size` / `--min-size` | `4` / `15` pixels |
| `--compression` | `zlib` |

Masks are inferred separately on each 2D plane and joined using `cellpose.utils.stitch3D`. This produces global IDs across z; it is not direct volumetric network inference.

Default label folder:

```text
cellpose_label_plane_s4_cp-1p75_f0p8_d90_n500_st0p25
```

### Cleanup defaults

The cleaner expresses size criteria in physical units and converts them to pixels/planes for each experiment.

| Criterion | Physical/default value | At 64 nm XY | At 74 nm XY |
| --- | --- | --- | --- |
| Minimum maximum nuclear cross-section | 12.288 µm² | 3,000 pixels | 2,244 pixels |
| Minimum sampled z-thickness | 0.722 µm | 2 planes at 361 nm z | 2 planes at 500 nm z |
| Maximum area of a small fragment label | 12.288 µm² | 3,000 pixels | 2,244 pixels |
| Minimum connected fragment-cluster area | 6.144 µm² | 1,500 pixels | 1,122 pixels |
| Minimum labels in a fragment cluster | 4 | 4 | 4 |

The first two criteria remove unreliable global objects. The fragment-cluster rule removes flagged labels on individual planes, using 8-connected groups of adjacent small masks. Surviving global IDs are preserved. The optional border-sliver rule is disabled unless `--remove-small-border-slivers` is supplied. The z-thickness criterion is a sampled-plane count, not a measured continuous nuclear height.

## 6. Run the v4.0 detector and Cellpose branch

Run from the directory containing the scripts. These examples use the original configured data roots. Step 1 also serves the StarDist-label branch; for that branch, continue with Section 9 after obtaining matching StarDist labels. Steps 2–5 below describe the Cellpose implementation, which was not used for the manuscript analysis.

### Step 1 — Detect spots

After confirming the calibration in Section 5, run the detector for an experiment. Its default scan is experiment 304 only:

```bash
python batch_get_spots_threshold12.py \
  --experiment-dir '/Volumes/Backup Plus/Experiment_results/304_Analysis_results/Experiment' \
  --experiment-average-threshold 12 \
  --resume
```

Repeat for other experiments with their own `Experiment` paths and verified source calibration, or use existing verified spot arrays. Single-input outputs are written into the corresponding `565/results/` folder.

Outputs include `spots.npy`, `spots_post_decomposition.npy`, `spots_post_clustering.npy`, `clusters.npy`, `smFISHChannelArray_LoG.tif`, `spotPlot.tif`, and diagnostic plots/text. Logs and timestamped CSV/JSON summaries are written in the selected experiment directory. Individual image failures are logged and skipped; **the batch can exit with code 0 even when images failed**, so inspect its summary.

### Step 2 — Segment and stitch nuclei

Inspect sample discovery, then run the configured five experiments:

```bash
python 'cellpose_sam_nuclei_3d(1).py' --dry-run

python 'cellpose_sam_nuclei_3d(1).py' \
  --device mps \
  --resume \
  --verbose
```

Use `--sample-regex` to select a pilot sample or subset. Sample IDs have the form `304__6h_Incision__Image1`.

Each label folder contains `Nucleus_Labels_*.tif`, `plane_objects.csv`, `tracking_links.csv`, `objects_3d.csv`, `run_configuration.json`, `sample_qc.json`, and `_COMPLETE.json` after successful completion. Inspect label overlays and the z-span/tracking summaries before continuing.

### Step 3 — Clean segmentation artifacts

```bash
python clean_cellpose_fragment_artifacts.py --all-experiments --dry-run

python clean_cellpose_fragment_artifacts.py --all-experiments
```

The default destination is a sibling directory with `_cleaned` appended to the label-folder name. It contains cleaned label planes, `removed_global_objects.csv`, `removed_plane_fragments.csv`, `cleaning_by_plane.csv`, and `cleaning_summary.json`.

The assignment stage requires `cleaning_summary.json` with `status: completed`. A dry run reports removal counts without creating a completed cleaned stack.

### Step 4 — Assign spots in 3D

```bash
python 'reassign_smfish_3d(1).py' \
  --output-root '/Volumes/Backup Plus/Experiment_results/smFISH_cellpose_assignment' \
  --verbose
```

Every spatially valid spot is assigned to a cleaned nuclear label using voxel-calibrated distance to the nuclear mask. The default has **no maximum assignment distance**. Spots outside the label stack's XY/Z bounds are recorded as invalid. The intended spot input uses integer coordinates; fractional XY coordinates are rounded for the per-plane nearest-voxel candidate lookup.

Outputs include every surviving cleaned nucleus, including zero-count nuclei. Here, “cell” is an analysis unit defined by a segmented nucleus; the script does not segment whole-cell boundaries.

Use `--no-assignment-maps` or `--no-examples` when those optional outputs are unnecessary. Completed samples are automatically reused; this script has no `--resume` option. See Section 10 before changing inputs or parameters in an existing output root.

### Step 5 — Collect all-cell CSVs

Use the collector matched to the current 3D output layout:

```bash
python export_all_cell_results_including_zero.py \
  --input-root '/Volumes/Backup Plus/Experiment_results/smFISH_cellpose_assignment' \
  --output-dir '/Volumes/Backup Plus/Experiment_results/all_cell_results_including_zero'
```

It copies each experiment's `results/all_cells_including_zero_wide.csv` to `<experiment>_output_all_cells_including_zero.csv`. The earlier name `all_cells_wide_including_zero.csv` is also accepted. Existing destination CSVs are refreshed without changing their values.

The collector always attempts all five configured experiments. A single-experiment pilot will therefore report missing sources for the others; its directly generated wide table is already usable.

**After separate `--sample-regex` assignment batches:** rerun the 3D assignment over the full intended sample set, without that filter, before collecting tables. Completed samples are reused and the experiment-level tables are rebuilt. The current 3D aggregates contain the samples selected for that invocation; the collector does not discover additional sample folders.

## 7. Cellpose branch output reference

Paths below are relative to the 3D assignment `--output-root`.

| Output path | Meaning |
| --- | --- |
| `run_configuration.json` | Experiment roots, voxel sizes, parameters, assignment rule, and statistical settings |
| `<experiment>/results/samples/<condition>/<animal>/cell_counts.csv` | One row per retained nuclear ID, including zeros |
| `<experiment>/results/samples/<condition>/<animal>/spot_assignments_nearest_3d.csv` | One row per input spot, including validity, assigned ID, nearest nuclear voxel, and distance in µm |
| `<experiment>/results/samples/<condition>/<animal>/assignment_count_map_3d.tif` | Nuclear voxels painted with that nucleus's total assigned spot count |
| `<experiment>/qc/samples/<condition>/<animal>/sample_qc.json` | Spot accounting, cell counts, selected input files, and assignment-distance summaries |
| `<experiment>/qc/samples/<condition>/<animal>/COMPLETE.json` | Sample completion record |
| `<experiment>/results/all_cells_long.csv` | Combined per-cell table for the selected samples in that experiment |
| `<experiment>/results/all_cells_including_zero_wide.csv` | One sample per column, all retained cells |
| `<experiment>/results/nonzero_cells_wide.csv` | One sample per column, positive cells only |
| `<experiment>/results/biological_replicate_summary.csv` | Per-animal/image means, responding percentages, and spot totals |
| `<experiment>/results/mean_count_unpaired_welch_ttests.csv` | Incision/amputation comparisons of sample means |
| `<experiment>/results/cell_percentage_gt_1_5_10.csv` | Per-sample percentages above strict count thresholds |
| `<experiment>/results/cell_percentage_gt_1_5_10_unpaired_welch_ttests.csv` | Welch comparisons of those percentages |
| `<experiment>/results/kruskal_wallis_biological_replicates.csv` | Omnibus tests across conditions |
| `<experiment>/results/dunn_test_biological_replicates.csv` | Pairwise Dunn tests with Holm correction |
| `<experiment>/qc/sample_qc.csv` | Combined sample QC |
| `<experiment>/example/<condition>/<animal>/` | Five-view PNG examples and selection metadata |

Related boxplots and assignment-distance plots are also written. Failed samples are reported in `processing_errors.csv` under experiment QC and in `processing_errors_all.csv` at the output root when errors occur.

### Interpreting counts and statistics

- `transcript_count` is the number of assigned detected/decomposed spots; the pipeline does not independently verify that every spot is a single transcript.
- `global_cell_id` identifies a nucleus within a sample. Use `sample_id` together with it when combining samples.
- Wide tables contain real zeros for zero-count cells and blank padding where columns have different lengths. **Do not convert padding to zero.** Rows are not matched cells across animals.
- `mean_count_all_cells` uses all retained nuclei. `mean_count_nonzero_cells` uses only nuclei with counts `> 0`; it is undefined/NaN for an all-zero sample.
- Threshold percentages use all retained nuclei as the denominator. `> 1`, `> 5`, and `> 10` are strict thresholds; for example, `> 1` means at least 2 spots.
- Cellpose-branch tests operate separately by experiment, using per-sample summaries. Welch tests compare incision and amputation at matching times, with Holm adjustment across times within each metric. Kruskal–Wallis/Dunn tests compare condition groups, with Holm adjustment of Dunn comparisons within each metric. These are outputs implemented in this branch, not a description of the manuscript's final statistical methods.
- The code treats each animal/image folder as an independent replicate. If multiple folders represent the same animal, aggregate appropriately before interpreting inferential statistics.
- Assignment distance should be reviewed because every valid spot is assigned, including distant spots. QC reports median, 95th percentile, maximum distance, and fractions exceeding 5, 10, and 20 µm.

## 8. Optional Cellpose benchmarking and image examples

### Cellpose parameter benchmark

The benchmark reads one z-plane and an ROI, reusing network flows across mask-dynamics settings. It compares Cellpose parameter settings; it is not itself a quantitative StarDist-versus-Cellpose equivalence test. The author's broader comparison is described in the development history. For example:

```bash
python cellpose_threshold_benchmark.py \
  --input '/path/to/nuclear_stack.tif' \
  --fiji-z 25 \
  --roi 0 0 2048 2048 \
  --device mps \
  --smooth-radii 2 4 \
  --diameters 90 \
  --niters 500 \
  --flow-thresholds 0.6 0.8 \
  --cellprob-thresholds -2 -1.75 \
  --norm-blocks 0 \
  --tile-overlap 0.5
```

Replace the input path, z-plane, and ROI with values valid for your image. `--fiji-z` is **1-based**; ROI coordinates are **0-based X, Y, width, height**. Alternatives include `--interactive-roi`, `--whole-plane`, or the default centered ROI.

Start inspection with `threshold_comparison.png`, then review `labels_*.tif`, `overlay_*.png`, `benchmark_summary.csv`, and `benchmark_settings.json`. A timestamped output directory is created beside the input unless `--output` is supplied.

Benchmark defaults differ from the batch segmentation defaults: cell probability `-2`, flow threshold `0.6`, and tile overlap `0.1`. The benchmark loads the installed Cellpose default model; the batch segmentation script explicitly selects `cpsam_v2`. Record/check the model used when comparing results. The benchmark calls a private mask-reconstruction method, so retain the working Cellpose environment for reproducibility.

### Full z-stack examples per condition

```bash
python export_condition_examples_gt20.py \
  --assignment-root '/Volumes/Backup Plus/Experiment_results/smFISH_cellpose_assignment' \
  --output-root '/Volumes/Backup Plus/Experiment_results/smFISH_condition_examples_gt20' \
  --threshold 20 \
  --crop-size 512 \
  --random-seed 20260901
```

For each experiment/condition, this script draws replicate number 1–3, falls back downward if necessary, and selects a random cell with `transcript_count > 20`. If none qualifies, it chooses among maximum-count cells in the selected replicate. An additional first-available fallback handles unusual sample names.

It exports nine TIFF z-stacks: original smFISH, magenta smFISH, yellow spots, their merge, original nuclei, colored segmentation, nuclei/segmentation merge, raw uint32 assignment-count map, and assignment/spot merge. Raw intensity crops retain their values; colored visualization outputs use display scaling. Selection metadata and `selection_summary.csv` document the choices.

Both channels must have matching stack dimensions, labels must cover raw z-indices `0..Z-1`, and XY dimensions must support the requested crop. Raw-data roots and the cleaned-label folder are configured in this script's constants; the CLI does not override them.

These are selected expression examples, not unbiased population samples. The automatic PNG examples created during 3D assignment use a different rule: one crop per sample, a default `> 10` selection threshold with maximum-count fallback, and an smFISH maximum projection alongside a selected nuclear plane.

## 9. Version 4.0 StarDist projection-label workflows

These utilities preserve the projection-label route within v4.0. They consume existing `565/results/labels/Nucleus_Labels_*.tif` masks generated by the StarDist notebook, while providing batch assignment, explicit linking modes, and all-cell exports. They do not generate StarDist masks or use the cleaned Cellpose labels.

### Compare no linking and reciprocal linking

```bash
python reassign_smfish_old_labels_modes.py \
  --link-mode none \
  --output-root '/Volumes/Backup Plus/Experiment_results/smfish_old_labels_no_link_final'

python reassign_smfish_old_labels_modes.py \
  --link-mode reciprocal \
  --benchmark-linking \
  --output-root '/Volumes/Backup Plus/Experiment_results/smfish_old_labels_reciprocal_final'

python compare_smfish_linking_nonzero_sample_means.py \
  --linking-root '/Volumes/Backup Plus/Experiment_results/smfish_old_labels_reciprocal_final' \
  --no-linking-root '/Volumes/Backup Plus/Experiment_results/smfish_old_labels_no_link_final' \
  --output-root '/Volumes/Backup Plus/Experiment_results/smfish_linking_sensitivity_nonzero_sample_mean'
```

Projection defaults are `--projection-size 10`, `--expansion-distance 20` XY pixels, `--min-smaller-overlap 0.35`, and `--min-iou 0.15`. Verify that the projection size matches the original mask-generation workflow. Spots are mapped by rounded z-coordinate integer division by the projection size, then assigned within expanded XY labels. Spots outside those expanded masks remain background/unassigned.

With `--link-mode none`, each label in each projection is a separate analysis unit. Reciprocal mode links best-overlap partners across adjacent projections when smaller-object overlap is at least 0.35 **or** IoU is at least 0.15. `--benchmark-linking` saves candidate-link review information and red/green overlay crops.

The comparison script includes **304/306 for wnt1** and **314/332 for notum**, restricted to **6 h and 12 h incision/amputation**. Experiment 307 is excluded. It computes each sample's mean among positive cells, then performs two-sided Welch comparisons across sample means, reporting raw p-values, Holm-adjusted p-values, and Benjamini–Hochberg q-values within each mode. It expects three samples per experiment/time/injury group by default. `--allow-incomplete` relaxes the design-count check but does not allow all-zero positive-cell means or fewer than two sample means per tested group.

### Earlier temporary reciprocal-linking script

```bash
python reassign_smfish_projection_for_now.py \
  --output-root '/Volumes/Backup Plus/Experiment_results/smfish_reassignment_projection_for_now'
```

This script records `assignment_backend=projection_linking_temporary`. Use the explicit modes script above for the documented linking/no-linking comparison; the comparison tool checks backend metadata.

### Rebuild all-cell tables from compatible sample folders

```bash
python export_all_cells_including_zero.py \
  --input-root '/Volumes/Backup Plus/Experiment_results/smfish_old_labels_reciprocal_final'
```

This exporter scans for `cell_counts.csv` with **`COMPLETE.json` in the same folder**, reading adjacent `sample_qc.json`. By default it writes `all_cells_including_zero_export/`, including long/wide tables, per-experiment CSVs, `animal_summary.csv`, `activating cell percentage.csv`, and export QC. Its percentage thresholds are strictly `> 0`, `> 1`, `> 2`, `> 5`, `> 10`, and `> 100`.

**Compatibility:** despite its broad introductory docstring, this supplied exporter does not directly match the supplied 3D script's split `results/samples/` and `qc/samples/` layout. Pointing it at the current 3D root will skip completed samples under its default discovery rule. `--include-incomplete` bypasses that check but does not reconnect the separate QC files. Use the collector in Step 5 for current 3D results.

## 10. Custom paths, reruns, and completion checks

| Script/stage | Configuration or rerun behavior |
| --- | --- |
| Spot detector | `--experiment-dir` selects the `Experiment` directory. `--resume` checks completion status, image path, and detection threshold; it does not validate all source constants or image contents. Omit it when intentionally regenerating after such changes. Outputs use the same filenames. |
| Cellpose segmentation | Repeatable `--experiment 'NAME,GENE,ROOT'` replaces the default experiment list. `--resume` continues compatible partial outputs. Completed outputs require matching recorded parameters. Use a new `--output-name` for parameter variants. |
| Cleaner | Use `--input-dir` for a custom stack and `--experiment-id` or `--voxel-size-nm X Y Z` for calibration. `--output-dir` selects a fresh destination. All-experiment mode uses hard-coded roots, skips completed cleaned folders without comparing changed settings, and skips incomplete nonempty output folders. |
| 3D assignment | Repeatable `--experiment 'NAME,GENE,ROOT,X_NM,Y_NM,Z_NM'` replaces the default list. `--cleaned-label-dirname` selects a different cleaned folder. Completed samples are reused without validating changed inputs/parameters; choose a new output root for a new analysis or intentionally regenerate with `--overwrite`. |
| Projection assignment | Repeatable `--experiment 'NAME,GENE,ROOT_OR_EXPERIMENT_DIR'`; completed outputs require compatible version/parameters. Partial folders require intentional regeneration with `--overwrite`. |
| Condition-example exporter | `--assignment-root` and `--output-root` are configurable; update `DEFAULT_EXPERIMENT_ROOTS` and `CLEANED_LABEL_DIRNAME` in source when relocating raw data or changing label folders. |

For example, a custom 3D assignment run can be specified as:

```bash
python 'reassign_smfish_3d(1).py' \
  --experiment '304,wnt1,/path/to/304_Analysis_results,64,64,361' \
  --output-root '/path/to/new_assignment_output'
```

Changing the segmentation output name also requires selecting the corresponding input folder in the cleaner and the resulting cleaned folder in assignment. The folder name itself is not a substitute for the recorded settings.

Before interpreting a run, confirm that expected samples were discovered, label cleaning completed, spot accounting is consistent, and large assignment distances have been reviewed. If inputs changed, previously completed downstream results need deliberate regeneration. Preserve prior outputs in separate directories when comparing analyses.

Built-in synthetic checks are available for most stages. Examples:

```bash
python 'cellpose_sam_nuclei_3d(1).py' --self-test
python 'reassign_smfish_3d(1).py' --self-test
python export_all_cell_results_including_zero.py --self-test
```

The projection scripts, general all-cell exporter, linking comparison, and condition-example exporter also provide `--self-test`. The detector, cleaner, and benchmark do not. These checks exercise implemented logic; they do not validate segmentation or spot-detection accuracy on biological images.

## 11. Comparing versions and reproducing an analysis

| Aspect | v3.10 notebooks | v4.0 StarDist-label branch | v4.0 Cellpose branch |
| --- | --- | --- | --- |
| Execution | Manually configure and run one image at a time; threshold setup can scan multiple images | Batch detector and assignment scripts | Batch segmentation, cleanup, assignment, and exports |
| Spot processing | Big-FISH with optional control-intensity filtering; later simplified use generally omits it | Batch Big-FISH, fixed threshold 12 by default; can reuse prior arrays | Same detector or verified prior spot arrays |
| Nuclear segmentation | StarDist2D on z-projections | Reuses existing StarDist projection masks | Cellpose-SAM per raw z-plane plus stitching |
| Relationship across z | Projection-local IDs in the original assignment notebooks | Independent projection labels or reciprocal overlap links | Stitched global nuclear IDs |
| Spot assignment | Projection index and expanded XY labels | Projection index and one XY label expansion | Calibrated nearest cleaned nuclear voxel, with no distance cutoff |
| Zero-count labels | Present in saved count arrays; many analysis cells remove them | Explicit all-cell and positive-cell tables | Explicit all-cell and positive-cell tables |
| Interpretation for the manuscript | Source workflow for the StarDist-based analysis; exact run settings and output provenance still matter | Retains and extends that analysis route; verify which outputs were actually used | Benchmarked, but not used for manuscript analysis |

For an analysis record, retain the source image/sample identifiers, preprocessing and ROI choices, voxel sizes, detection threshold, control-filtering status, segmentation model, projection/stitching settings, label expansion/assignment mode, zero-cell handling, and the statistical unit. Preserve the output files that support each figure, together with the code/environment that generated them.

Use separate output roots for notebook reproduction, v4.0 no-linking and reciprocal-linking results, and Cellpose results. Similar benchmark outcomes do not make those configurations interchangeable. The current source snapshots also contain experiment-specific values: for example, notebook 1 points to an EGR1 dataset with `voxel_size = (601, 75, 75)`, whereas the batch detector uses `(361, 75, 75)`. Neither is a universal calibration for every dataset.

The code and historical instructions document available implementations. Publication provenance and the qualitative benchmark outcome above follow the author's account. No new image analysis or StarDist/Cellpose benchmark was performed during this documentation update.
