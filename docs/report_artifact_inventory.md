# Report artifact inventory and drafting template

## Report readiness

Yes. The repository has enough evidence for a technical/project report on the maintained EEG-image experiments: executed report notebooks, post-hoc metric CSV/YAML files, selected checkpoints, TensorBoard event files, configs, local launchers, and qualitative images. The 12-run EEG alignment sweep is the central controlled result; prior reconstruction, CoMM, text alignment, and low-level reconstruction are supporting tracks.

This is not evidence for an exact reproduction of external EEG-guided diffusion papers. The repository reconstruction path omits paper-specific components, and the secondary tracks are not matched multi-seed studies. State those limits directly.

## Evidence rules

- `test_metrics.csv` and adjacent `evaluation_config.yaml` are the source of truth for test metrics and checkpoint provenance.
- `aggregated_metrics.csv` is the source of truth for seed summaries in the alignment sweep.
- Use executed notebooks for tables, plots, and their recorded artifact paths—not as independent evidence beyond those artifacts.
- Label the second-prior work historical. Do not fold it into maintained-method results.
- Do not compare total training losses across different objectives. The low-level MSE control is L2-only, whereas SynCLR variants add a perceptual term.

## Maintained artifact map

| Track | Evidence and reproducibility path | Visual assets | Status |
|---|---|---|---|
| EEG-image alignment | `../experiments/eeg_alignment_sweep/aggregated_metrics.csv`; 12 run-level CSV/YAML pairs; `../src/brain_image/configs/train_eeg_align.yaml`; `../scripts/run_eeg_alignment.sh`; `../scripts/evaluation/test_eeg.py` | 216 target/chosen PNGs: 9 examples x 12 runs x 2 images | Executed: `../notebooks/eeg_alignment_sweep_report.ipynb` |
| Prior reconstruction | `../experiments/eeg_prior/20260831_134809/train_eeg-20260831_134844/version_0/test/`; `train_eeg_prior.yaml`; `run_eeg_prior.sh` | 9 GT/reconstruction pairs | Executed: `../notebooks/eeg_prior_reconstruction_report.ipynb` |
| Joint alignment + prior | `../experiments/eeg_alignprior/20260904_100813/train_eeg-20260904_100911/version_0/test/`; `train_eeg_alignprior.yaml`; `run_eeg_alignprior.sh` | 9 alignment target/chosen pairs and 9 GT/reconstruction pairs | Same executed prior report |
| CoMM fusion | `../experiments/experiments/comm/20260901_084821/train_comm-20260901_084858/version_0/test/`; `train_comm.yaml`; `run_comm.sh`; `test_comm.py` | No full-run image artifacts | Executed: `../notebooks/comm_report.ipynb` |
| Text alignment | `../data/things-eeg2/captions/qwen3.5-0.8b.jsonl`; `../tensorcache/qwen3.5-0.8b/t5_base/`; `../experiments/eeg_alignment_text/20260904_002351/train_eeg-20260904_002425/version_0/test/`; `train_eeg_align_text.yaml` | No dedicated qualitative export | Executed: `../notebooks/text_alignment_report.ipynb` |
| Low-level reconstruction | Three run-level CSV/YAML pairs under `../experiments/low_level/`; `train_low_level*.yaml`; `run_low_level.sh`; `test_low_level.py` | Five matched test examples for each of MSE, unaligned SynCLR, aligned SynCLR | Executed: `../notebooks/low_level_comparison_report.ipynb` |
| Second prior | Commits `44f8854` (introduced) and `2cca62a` (removed); archived `../experiments/prior_second/` | Historical only | Executed: `../notebooks/second_prior_history.ipynb` |

The `experiments` directory is a storage symlink to `/storage/experiments/brain-image-implementation`. Follow it when validating assets. The nested CoMM path is real legacy provenance; do not silently rewrite it.

## Quantitative anchors

### Main alignment study

The 12-run alignment sweep uses three seeds for each image-target/EEG-encoder condition. Its primary metric is EEG-to-image top-1 accuracy (`image_acc`).

| Image target | EEG encoder | `image_acc`, mean +/- std | `brain_acc`, mean +/- std | Similarity, mean +/- std |
|---|---|---:|---:|---:|
| Aligned SynCLR ViT-B/16 | ATMS | 0.5983 +/- 0.0252 | 0.6450 +/- 0.0260 | 0.2823 +/- 0.0044 |
| Aligned SynCLR ViT-B/16 | NICE | 0.5283 +/- 0.0419 | 0.6033 +/- 0.0126 | 0.2702 +/- 0.0023 |
| Unaligned SynCLR ViT-B/16 | ATMS | 0.5450 +/- 0.0304 | 0.5733 +/- 0.0153 | 0.2985 +/- 0.0058 |
| Unaligned SynCLR ViT-B/16 | NICE | 0.5067 +/- 0.0126 | 0.5550 +/- 0.0200 | 0.2872 +/- 0.0023 |

Best individual run: aligned SynCLR + ATMS, seed 42, with `image_acc=0.625`, `brain_acc=0.660`, similarity `0.2861`.

### Prior and joint prior

| Variant | Prior cosine | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | InceptionV3 | CLIP | EfficientNet | SwAV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prior-only | 0.6340 | 0.0972 | 0.3086 | 0.7222 | 0.8889 | 0.6389 | 0.7917 | 0.9004 | 0.5880 |
| Joint alignment + prior | 0.6420 | 0.1811 | 0.3157 | 0.9444 | 0.9167 | 0.8056 | 0.8472 | 0.8912 | 0.5645 |

The joint run also reached `brain_acc=0.595`, `image_acc=0.515`, similarity `0.2642`.

### CoMM fusion: negative result

Use retrieval against the ground-truth target image latent, not the training-time generated-branch diagnostics.

| Representation | Query-to-image top-1 | Image-to-query top-1 |
|---|---:|---:|
| EEG-only branch | 0.235 | 0.250 |
| Generated-prior-only branch | 0.050 | 0.095 |
| Fused CoMM prototype | 0.145 | 0.190 |

2026-10-14 direction audit: the CoMM CSV's named `acc_*_to_target_img` keys contain column-wise image-to-query accuracy, while its `acc_target_img_to_*` keys contain row-wise query-to-image accuracy. The table above and corrected `comm_report.ipynb` follow matrix axes, rather than those reversed key names. The notebook now resolves the nested archive and preserves recorded/current checkpoint paths. The main alignment/text metrics use a different naming assignment and are unchanged.

2026-10-14 scope decision: C01 has been dropped from report-completion work at the user's direction. This report is completed from existing evidence only, with at most notebook reruns; no retraining or expanded evaluation is required. See `docs/report_completion_map.md` for the accepted C02--C06 scope.

### Text-target alignment

- 16,740 non-empty Qwen3.5 captions: 16,540 train and 200 test images.
- Cached T5-base embeddings: 16,540 train and 200 test tensors.
- Selected checkpoint: `epoch_0029-val_align_top1_0.5250.ckpt` after early stopping at epoch 129.
- Test: `image_acc=0.170`, `brain_acc=0.215`, similarity `0.2161`.

### Low-level MSE/SynCLR comparison

| Variant | PixCorr | SSIM | AlexNet-2 | AlexNet-5 | InceptionV3 | CLIP | EfficientNet | SwAV | Aligned DreamSim | Unaligned DreamSim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MSE | 0.2431 | 0.3669 | 0.7016 | 0.6965 | 0.5479 | 0.5162 | 0.9739 | 0.7549 | 0.7877 | 0.5579 |
| Unaligned SynCLR | 0.1976 | 0.4171 | 0.7316 | 0.7117 | 0.5425 | 0.5525 | 0.9868 | 0.7211 | 0.7216 | 0.5438 |
| Aligned SynCLR | 0.1859 | 0.4085 | 0.7905 | 0.7315 | 0.5614 | 0.5381 | 0.9886 | 0.7237 | 0.6886 | 0.5584 |

No low-level variant dominates every metric.

## Notebook classification

### Primary evidence

- `eeg_alignment_sweep_report.ipynb`
- `eeg_prior_reconstruction_report.ipynb`
- `comm_report.ipynb`
- `text_alignment_report.ipynb`
- `low_level_comparison_report.ipynb`
- `second_prior_history.ipynb` — context only, not a maintained-method result

`low_level_report.ipynb` is a valid single-baseline supplement; the comparison notebook supersedes it for conclusions and figures.

### Do not use as primary report evidence without revalidation

- `qwen_text_embedding_report.ipynb` uses earlier `local.jsonl` / Qwen2.5 defaults; use the Qwen3.5/T5 artifacts and text-alignment report for the maintained result.
- `comm.ipynb`, `comm_prior.ipynb`, `diffusion_prior.ipynb`, `low_level.ipynb`, `text_embedding.ipynb`, and exploratory notebooks.
- `scratch.ipynb`, `tmp.ipynb`, `try_acc.ipynb`, and similar ad-hoc notebooks.

## Report outline

1. **Abstract** — problem, maintained workflow, strongest controlled alignment result, and carefully scoped reconstruction findings.
2. **Problem and data** — Things-EEG2, subject/split scope, frozen target encoders, and data provenance.
3. **Method** — alignment, diffusion prior, CoMM, text targets, and low-level VAE-latent decoding as distinct tracks.
4. **Protocol and reproducibility** — Hydra, Lightning, training-only TensorBoard, separate CSV/YAML evaluation, checkpoint selection, and seed policy.
5. **Results** — lead with alignment sweep; then prior, CoMM, text, and low-level ablation.
6. **Discussion and limitations** — metric disagreement, negative CoMM result, one-seed secondary tracks, and no external-paper parity claim.
7. **Appendix** — command/config/checkpoint/CSV manifest and metric definitions.

## Figure templates

### F1 — Method overview

```text
EEG x -> EEG encoder f_theta -> aligned latent z_E -> retrieval against frozen targets
                              -> diffusion prior p_phi -> generated latent -> reconstruction
                              -> low-level head h_psi -> frozen VAE decoder -> 128px image
CoMM: z_E + generated prior latent -> fusion module -> prototype z_C
```

> Figure 1. Overview of the maintained EEG-image pipelines. Solid arrows are evaluated paths; frozen target encoders and decoders are shown as fixed modules. `[Confirm encoder names and dimensions in the final artwork.]`

### F2 — Main alignment sweep

**Source:** `experiments/eeg_alignment_sweep/aggregated_metrics.csv`.

**Template:** grouped dot/bar plot of `image_acc`; each seed is a point and each condition shows mean +/- standard deviation. Add a secondary panel for `brain_acc`.

> Figure 2. Held-out top-1 retrieval across 12 alignment runs. Points are seeds 41–43; the primary direction is EEG-to-image retrieval. `[Insert exact held-out test size and subject scope.]`

### F3 — Qualitative alignment retrieval

**Source:** `experiments/eeg_alignment_sweep/<run>/.../test/align/{target,chosen}.png`.

**Template:** three rows for the top-three `image_acc` runs; three target/chosen pairs per row. Label encoder, target, seed, and run-level score.

> Figure 3. Qualitative held-out retrievals for the top alignment runs. These fixed examples illustrate behavior; the aggregate metrics in Figure 2 remain the quantitative comparison.

### F4 — Prior reconstruction comparison

**Source:** prior-only and joint run `test/*/prior/{gt,recon}.png` files.

**Template:** rows are common held-out stimuli; columns are ground truth, prior-only reconstruction, joint reconstruction.

> Figure 4. Reconstructions from the repository's simplified diffusion-prior path. They illustrate this implementation and are not a direct reproduction of the full external EEG-guided diffusion pipeline.

### F5 — CoMM result

**Source:** full CoMM `test_metrics.csv`.

**Template:** query-to-image bar chart: EEG-only `0.235`, generated-prior-only `0.050`, fused CoMM `0.145`. An optional reverse-direction panel uses `0.250`, `0.095`, and `0.190` respectively.

> Figure 5. CoMM fusion did not improve ground-truth image-latent retrieval over EEG-only retrieval in the evaluated full run.

### F6 — Text-target pipeline

**Template:** stimulus image -> Qwen3.5 caption -> frozen T5 embedding, jointly shown with EEG -> encoder -> same text-embedding space. Add caption/embedding counts and test retrieval metrics.

> Figure 6. Text-target alignment uses Qwen3.5 captions and frozen T5-base embeddings. `[Only show caption examples after verifying their prompt/model provenance.]`

### F7 — Low-level ablation

**Source:** `notebooks/low_level_comparison_report.ipynb` and saved `test/reconstructions/example_{000,050,100,150,199}.png`.

**Template:** five rows of fixed examples; columns: ground truth, MSE, unaligned SynCLR, aligned SynCLR. Pair with a selected-metric plot or heat map.

> Figure 7. Matched low-level reconstructions under the same frozen VAE decoder. The metrics show trade-offs rather than a single uniformly best perceptual loss.

## Equation templates

### E1 — Symmetric InfoNCE

```latex
z_i^E = \frac{f_\theta(x_i)}{\lVert f_\theta(x_i) \rVert_2},
\qquad
z_i^T = \frac{g(I_i)}{\lVert g(I_i) \rVert_2},
```

```latex
\mathcal{L}_{\mathrm{sym}} = \frac{1}{2N}\sum_i
\left[-\log\frac{e^{(z_i^E)^\top z_i^T/\tau}}{\sum_j e^{(z_i^E)^\top z_j^T/\tau}}
-\log\frac{e^{(z_i^T)^\top z_i^E/\tau}}{\sum_j e^{(z_i^T)^\top z_j^E/\tau}}\right].
```

**Explanation placeholder:** The trainable EEG encoder is aligned to a frozen image or text target encoder with a bidirectional contrastive objective. `[Confirm temperature and normalization from resolved config.]`

### E2 — Top-1 retrieval

```latex
\mathrm{Acc}_{E\to T} = \frac{1}{N}\sum_i
\mathbf{1}\left[i = \arg\max_j (z_i^E)^\top z_j^T\right].
```

**Explanation placeholder:** `image_acc` is EEG-to-image top-1 and the primary alignment metric; `brain_acc` is the reverse-direction diagnostic.

### E3 — Prior objective

```latex
\mathcal{L}_{\mathrm{prior}} =
\lambda_{\mathrm{diff}}\mathcal{L}_{\mathrm{diff}}
+ \lambda_{\mathrm{sim}}(1-\cos(\hat z^I,z^I))
+ \lambda_{\mathrm{Huber}}\mathrm{Huber}_{\delta}(\hat z^I,z^I)
+ \lambda_{\mathrm{align}}\mathcal{L}_{\mathrm{contrast}}(\hat z^I,z^I).
```

**Required finalization:** replace the diffusion term and weights with the resolved configuration of each reported checkpoint.

### E4 — CoMM fusion

```latex
z_i^C = C_\psi(z_i^E, \hat z_i^I),
\qquad
\mathrm{Acc}_{C\to I} = \frac{1}{N}\sum_i
\mathbf{1}\left[i = \arg\max_j (z_i^C)^\top z_j^I\right].
```

**Explanation placeholder:** CoMM fuses EEG and generated-prior embeddings; compare its prototype against the ground-truth image target alongside the EEG-only and prior-only baselines.

### E5 — Low-level reconstruction

```latex
\hat z_i^L=h_\psi(f_\theta(x_i)),
\qquad \hat I_i=D_{\mathrm{VAE}}(\hat z_i^L),
```

```latex
\mathcal{L}_{\mathrm{low}}=\lVert\hat z_i^L-z_i^L\rVert_2^2
+\lambda_{\mathrm{perc}}d(q(\hat I_i),q(I_i)).
```

**Required finalization:** for SynCLR, (q(I_i)) is the cached aligned or unaligned target. For the MSE control, set \(\lambda_{\mathrm{perc}}=0\).

## Reusable writing templates

### Results paragraph

> On the held-out test set, **[method]** achieved **[metric=value]**. Across **[number]** seeds, the mean was **[mean +/- std]**, compared with **[baseline mean +/- std]**. This supports **[narrow conclusion]** for **[subject/data/configuration]**; it does not establish **[out-of-scope broader claim]**.

### Negative-result paragraph

> The proposed **[extension]** did not improve **[primary metric]** over **[baseline]** in the evaluated run (**[value]** versus **[value]**). We treat this as a negative result for the present configuration, not as evidence that the class of methods is universally ineffective.

### Qualitative-results paragraph

> Figure **[N]** uses fixed held-out samples selected by **[rule]**. The images illustrate model behavior; the aggregate metrics in Table/Figure **[N]** remain the quantitative basis for comparison.

### Limitation paragraph

> These reconstruction scores use the repository's simplified evaluation path and are not a direct reproduction of **[external paper]**. Because **[track]** has **[number]** completed seeds, this comparison is descriptive rather than statistically powered.

## Final report gaps

- Create missing `docs/project.md` and `docs/architecture.md`, or explicitly replace that repository documentation requirement with the final report.
- Verify the Things-EEG2 citation, preprocessing wording, license, subject count, and held-out split before publication.
- Caption JSONL records do not preserve per-record prompt/model provenance; verify this before showing captions.
- Generate a release manifest with source commit, lockfile hash, checkpoint SHA256, and final artifact paths for durable claims.
- Keep old `encoders*`, `modes`, `variance`, `subs`, and other archived sweep directories out of the main report unless their protocol is separately verified.
- Do not claim low-level loss-objective parity from total loss values, and do not claim paper parity from current reconstruction scores.

## Pre-writing checklist

- [x] Main multi-seed alignment metrics and retrieval images.
- [x] Prior-only/joint reconstruction metrics and image pairs.
- [x] CoMM ground-truth-target comparison.
- [x] Text artifact counts and text-alignment metrics.
- [x] Low-level three-way metrics and matched image grid.
- [x] Historical second-prior narrative.
- [ ] Dataset/paper citations and preprocessing wording checked.
- [ ] Formal environment/checkpoint manifest generated.
- [ ] Final claim review against the scope notes above.
