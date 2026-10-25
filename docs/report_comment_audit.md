## Section 3 separation of theory, methods and results, 2026-10-25

- Implemented the user-approved four-part outline after removing the repeated representation-learning introduction: 3.1 motivation/questions, 3.2 reconstruction theory, 3.3 local methods/evaluation, and 3.4 results/discussion.
- Theory starts with the existing brain representation and explains the prior-to-generator workflow. DALL-E 2 and modality gap form one supporting subsubsection, followed by low-level reconstruction. Removed SDEdit discussion and completed drafting TODOs from section 3.
- Sources read: MindEye Sections 2.1--2.2 and Appendix A.3 (https://arxiv.org/html/2305.18274); Visual EEG Sections 2.3--2.4 and Appendix C (https://arxiv.org/html/2403.07721v5); DALL-E 2 Section 2.2 (https://arxiv.org/html/2204.06125); Mind the Gap (https://arxiv.org/html/2203.02053); BrainBits Sections 3--7 (https://arxiv.org/html/2411.02783v1). Added the Liang et al. bibliography entry.
- Corrected an outline error using the active code path: EEGAlignmentModel instantiates SimpleDiffusionPrior, an MLP encoder-decoder with skip connections, not the older transformer prior in the same file. Its forward output predicts noise and remove_noise derives the clean estimate used by the three-term objective. Image generation uses SDXL-Turbo with the ViT-H IP-Adapter, as verified in reconstruction.py. No model changes.
- Preserved the original three tables, two figures, their captions and labels, and both existing local loss equations exactly. All original reported decimal values remain; secondary joint retrieval directions are explicit. Removed duplicate preliminary results and separated 200-target prior evaluation from nine-image reconstruction metrics.
- Expanded BrainBits discussion to cover 50-dimensional bottlenecks, CLIP/SSIM/PixCorr preservation, Gaussian-input baseline, oracle-latent ceiling and the distinction from shuffled EEG. Kept dimensionality distinct from bits and fMRI findings distinct from local EEG evidence. Consolidated control proposals and added the controlled low-level comparison to Future Ideas.
- Verification: final 18-page LuaLaTeX build has no undefined citations/references, fatal errors, overfull or underfull boxes. Inspected section 3 on pages 5--13, Future Ideas on 16--17 and references on 17--18. Source checks confirm exact tables/figures/local losses, unchanged sections 1/2/4, unique labels, all active references/citations resolved and no newly introduced whitespace errors. Existing unrelated section-2 trailing whitespace retained.
- No training, inference, experiment, API or raw evaluation changes; no README changes. The requested docs/project.md and docs/architecture.md remain absent.

## ATM-S attribution and human-alignment comparison, 2026-10-25

- Read Li et al., Visual Decoding and Reconstruction via EEG Embeddings with Guided Diffusion, https://arxiv.org/html/2403.07721v3, especially Section 2.1, Section 3.2 and Appendix B.3. Inspected the local atms.py and its upstream attribution. Attention, spatiotemporal convolution, temporal aggregation and projection are inherited architecture, not a local contribution.
- Corrected 2.2 to attribute NICE to Song et al. and ATM-S to Li et al.; separated local output configuration, cached targets and training.
- Removed internal metric keys throughout the report, replacing two additional occurrences in reconstruction and caption-alignment summaries with the correct directional retrieval names. All numerical values retained.
- Focused results on human-aligned versus unaligned targets within each EEG encoder. Removed encoder-ranking commentary, qualitative-results prose and the single-condition retrieval illustration as requested. Preserved all eight mean/standard-deviation values in the results table and the seed-comparison figure.
- Closed with the requested agreement statement about AlignVis. Section 2.1 unchanged.
- Verification: 17-page LuaLaTeX build, no undefined citations/references or overfull boxes; inspected pages 3--6 and 14. Verified absent metric keys/qualitative section and unchanged numerical table values.

## Complete rewrite after section-boundary review, 2026-10-25

- Re-read AlignVis Sections 2 and 4.4: https://arxiv.org/html/2502.03081v3#S2 and https://arxiv.org/html/2502.03081v3#S4.SS4. The paper uses symmetric contrastive training and reports EEG-to-image top-1/top-5 retrieval over all 200 test images. Bidirectional training alone is not evidence that the paper reports reverse retrieval results.
- Rewrote 2.1 in third person throughout, retaining the technical formulation and metric definitions without local implementation conventions, metric keys or matching-index commentary. Described the loss as equivalent up to constant scaling and explicitly identified the EEG-to-image direction as the paper's reported scores.
- Rewrote 2.2 in six explicit stages: scope/comparison, implemented models, training, evaluation, quantitative results, qualitative results. All equations, tables and figures are unchanged; subsequent report sections are unchanged. Added the NICE citation and kept all numerical findings.
- Verified the absence of first-person pronouns and code identifiers in 2.1 and checked structural/content invariants. LuaLaTeX compiled 18 pages without undefined references/citations or overfull boxes. Visually inspected pages 2--6.

## Section 2 reorganization, 2026-10-25

- Implemented the user-approved separation: 2.1 contains the paper motivation, shared embedding/InfoNCE method, retrieval definitions, findings and reconstruction bridge; 2.2 contains the subject-8 scope, local models, training/evaluation choices and results.
- Preserved every equation and label, all result values, citations, figures and unrelated sections. Explicitly distinguished the report's bidirectional loss averaging from attribution to the paper, and human perceptual alignment from EEG-image alignment.
- Consolidated repeated settings and added a visible explanation that the three qualitative examples are illustrative.
- Verified equation bodies and results table/prose against the original source. LuaLaTeX compilation passed with no undefined references/citations, fatal errors or overfull boxes. Visually inspected affected pages 2--6.
- Environment: redirected the LuaTeX cache to a writable work directory; enabled shell escape required by the existing minted package. No experiment, metric or production code changes.

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
