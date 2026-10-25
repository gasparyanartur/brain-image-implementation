## Paper synopsis correction, 2026-10-25

- Replaced the human-aligned image encoder background with the approved two-paragraph synopsis. Removed the extended DreamSim/LoRA tutorial and gLocal/Harmonization comparison from the synopsis; preserved the separate recreation and its results.
- [Rajabi et al., Sections 1–2](https://arxiv.org/html/2502.03081v3#S2): the hypothesis is that human similarity judgments capture early impressions and align more closely with brain signals. The authors motivate this with rapid presentation (100 ms in EEG) and shape, orientation, and color; this is their proposed explanation.
- [Sections 4.2–4.4](https://arxiv.org/html/2502.03081v3#S4): use human-aligned vision models from Sundaram et al. (2024), compare against original pretrained versions, freeze image encoders, train brain encoders with InfoNCE, and evaluate top-1/top-5 retrieval against all 200 unseen test images. Appendix B states that aligned image models are obtained from their authors without retraining.
- [Section 5.2, Table 1](https://arxiv.org/html/2502.03081v3#S5.SS2): human-aligned targets improve within-participant retrieval across the tested image and EEG architectures. This statement does not imply uniform cross-subject benefits: Section 5.3 reports significant cross-subject improvements only for CLIP and OpenCLIP.
- [Section 5.5](https://arxiv.org/html/2502.03081v3#S5.SS5): gradient analysis emphasizes earlier responses (0–200 ms) and Alpha/Beta/Gamma components for human-aligned models. Both variants use occipital/parietal information, without a significant spatial difference. Treat the temporal/spectral findings as supporting observations, not causal proof of the proposed mechanism.
- [Section 6](https://arxiv.org/html/2502.03081v3#S6): conclusion emphasizes human-aligned representations and experimental design. Future work proposes reconstruction with human-aligned conditioning, quantifying alignment, validation on brain responses to other image databases, and fMRI with higher spatial resolution/longer presentations. Appendix E of this version already contains an fMRI evaluation, despite the prospective wording in the discussion.
- The reconstruction proposal supplies the connection to our project. Our implementation and results remain separately attributed under “Our recreation” and the reconstruction sections.

# Report comment iteration, 2026-10-13

## Scope review, 2026-10-14

The user restricted completion to existing evidence and at most notebook reruns. C01 is dropped; its previous report paragraph and proposed split-repair work are removed. Historical audit observations below remain a record of the earlier iteration, not current completion requirements.

C02/C03 are now resolved in the executed CPU-only CoMM notebook: corrected direction mapping/table/plot, current nested archive, recorded/resolved checkpoint paths, and SHA256. Raw metric CSVs and evaluator code are unchanged. C05 is isolated by training/validation/inference input/reference tables and a separate internal-diagnostic table; no causal intervention was performed. C04 keeps the nine-image reconstruction scope. C06 is documented through saved-config budget/batch/selected-epoch values and a future-work proposal, with no training.

## Follow-up iteration

- Implemented the five new AGENT points: expanded the compact-conditioning/generator-prior conclusion; removed the SigLIP subsection as requested; added CoMM's paper definition and the local fusion hypothesis; and added six future-work proposals.
- Read [Dufumier et al., CoMM](https://arxiv.org/html/2409.07402v1), Sections 2--3, and added its ICLR 2025 bibliography entry. The text distinguishes the paper's multimodal objective from our EEG-derived/generated-latent adaptation and explains the train/evaluation input shift.
- Corrected the report and inventory's CoMM direction labels by inspecting `get_retrieval_accuracy` and `get_target_retrieval_metrics`. Row-wise query-to-image values are 0.235/0.050/0.145; column-wise reverse values are 0.250/0.095/0.190. Original CSVs are unchanged. The supporting notebook still needs this correction; the previous iteration's forward labels below are superseded by this audit.
- Identified shared validation/test data-bank use in `ThingsEEG2Dataset` and disclosed it in the report's evaluation scope. The dataset file's February 2026 version already contains the same split routing, and the maintained trainer passes these loaders to Lightning without creating a separate validation partition.
- Added `docs/report_completion_map.md`: P0 correctness/protocol work, section-level deliverables and acceptance criteria, evidence sources, reproducibility/citation/figure tasks, ordered writing passes, and the distinction between a comprehensive historical report and stronger independent-test claims.

Implemented the current AGENT comments in `report/69bd47171c8bd99594e793c6/main.tex`, preserving section order and original TODO markers. Completed TODOs retain an addressed annotation. No training runs or production code were changed.

## Comment outcomes

| Request | Implementation/evidence |
| --- | --- |
| Subject-8 scope in introduction and alignment summary | Saved subject-8 hparams/evaluation configs; configurable subject/seed launchers. Subject variation attributed to Rajabi's participant results, not inferred from the three-seed local sweep. |
| Exact human-aligned SynCLR/DINOv2 procedure | Sundaram 2024, Sections 3.1--3.4 and Appendix C.1: NIGHTS preference triplets, cosine-distance margin loss, LoRA. |
| NICE citation | Song 2024, Section 3: Natural Image Contrast EEG and temporal-spatial convolution. |
| Table typography, alignment caption, retrieval model | Mathematical accuracy headings; concise condition/subject/metric caption; ATMS with human-aligned SynCLR ViT-B/16 identified from the generating notebook checkpoint. |
| Reconstruction citations and Huber equation | Read DALL-E 2 and SDEdit methods, MindEye2 low-level and inference sections, Li's ATM and reconstruction sections. Equation interleaved with method prose and corrected to match the three implemented prior loss terms. |
| Prior model/performance table and figures | Two saved test CSVs, selected checkpoint hparams, and nine evaluator image pairs per run. Table labels full-200 prior cosine versus nine-image reconstruction metrics. Three matched visual rows placed under MindEye and Visual EEG. |
| Low-level bullets/table/visualizations | Three saved test CSV/YAML pairs, low-level training step, frozen perceptual loss code, and executed comparison notebook. Added method/evaluation bullets, metric directions, seven metric rows, and five matched ground-truth/prediction rows. |
| BrainBits note | Read Mayo 2024 introduction and results; added fMRI-specific bottleneck finding and proposed EEG baselines without treating fMRI evidence as a measured EEG result. |
| SigLIP experiments | Found maintained loss implementation, `notebooks/losses.ipynb`, and `scripts/params/align_losses.json`. Followed the experiments symlink and searched archived YAML/JSON configs: no selecting SigLIP run with final test metrics found. Report states this gap. |
| Text alignment expansion | Saved caption JSONL/YAML, T5 encoder's masked mean pooling, text run hparams and CSV, maintained notebook. Added generation protocol, retrieval task, result table, and native TikZ pipeline. Kept the per-record provenance TODO; only generation-level metadata is present. |

## Primary sources read and claim checks

- [Perceptual alignment](https://arxiv.org/html/2410.10817v1): Sections 3 and Appendix C.1 support the specific human-aligned backbone procedure.
- [NICE](https://arxiv.org/html/2308.13234v3): Section 3 supports the framework name, contrastive objective, and convolutional encoder description.
- [EEG-guided diffusion](https://arxiv.org/html/2403.07721v3): Section 2.1, discussion, and Appendix C support ATM and two-stage reconstruction; the low-level VAE extension is discussed separately.
- [MindEye2](https://arxiv.org/html/2403.11207v2): Sections 2.2.3 and 2.5 support the low-level branch and image-to-image refinement description.
- [DALL-E 2](https://arxiv.org/html/2204.06125v1): Section 2 supports the prior/decoder separation and CLIP-latent conditioning.
- [SDEdit](https://arxiv.org/html/2108.01073v1): Sections 3--4 support perturbing an initial image then denoising with a prior. Final author metadata checked against the [revised record](https://arxiv.org/abs/2108.01073).
- [BrainBits](https://arxiv.org/html/2411.02783v1): Introduction and results support the fMRI bottleneck example and baseline/ceiling/bottleneck evaluation proposal.
- [SigLIP](https://arxiv.org/html/2303.15343v1): Section 3 supports pairwise logistic matching rather than batch-softmax normalization.

## Artifact provenance

`notebooks/report_reconstruction_figures.ipynb` reads saved predictions only, validates selected checkpoint existence and subject-8 configs, and exports `prior_comparison_grid.png` and `low_level_comparison_grid.png` directly into the report's `images/` directory. It checks that prior-run ground truths agree. Low-level ground truths use the sorted concept/image ordering implemented by `get_image_paths`; positions are 0, 50, 100, 150, and 199.

Metric sources:

- Prior-only: `experiments/eeg_prior/20260831_134809/train_eeg-20260831_134844/version_0/test/test_metrics.csv`.
- Joint: `experiments/eeg_alignprior/20260904_100813/train_eeg-20260904_100911/version_0/test/test_metrics.csv`.
- Low-level MSE: `experiments/low_level/20260905_025455/train_low_level-20260905_025524/version_0/test/test_metrics.csv`.
- Low-level unaligned: `experiments/low_level/unaligned_cached_20260905/train_low_level-20260905_180009/version_0/test/test_metrics.csv`.
- Low-level aligned: `experiments/low_level/aligned_cached_20260906/train_low_level-20260906_140857/version_0/test/test_metrics.csv`.
- Text: `experiments/eeg_alignment_text/20260904_002351/train_eeg-20260904_002425/version_0/test/test_metrics.csv`.

The prior loss is checked against `EEGAlignmentModel._run_step_do_recon` and each selected run's `hparams.yaml`: cosine weight 1, Huber weight 0.1, contrastive weight 0.1, Huber delta 0.5. There is no additional separately summed diffusion MSE term in that step.
