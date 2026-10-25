# Remaining report work

Use existing results and notebooks only. No retraining or expanded evaluations. Keep the report's structure and original TODOs.

## Introduction
- State the project questions, what we implemented, and the main findings.
- Make the subject-8 scope and recreated-versus-extended work clear.
- Write a short abstract once the rest is finished.

## Background and alignment
- [x] Keep dataset context brief: subject 8 and the reused THINGS-EEG2 preprocessing.
- [x] Explain NICE versus ATMS and aligned versus unaligned image targets.
- [x] Add the important training settings and interpret the existing alignment table/plot.
- [x] Make the sample counts, retrieval directions, and figure-selection rule clear.

## Reconstruction
- Explain the actual prior and low-level pipelines, including what is trained or frozen.
- Name the target spaces, generator, losses, and sampling settings.
- Interpret the prior-only/joint results using the existing nine-image evaluation.
- Explain the low-level metric trade-offs and visible reconstruction limitations.
- Summarize what the generator contributes and what remains unknown about information recovered from EEG.

## Retrieval
- CoMM: finish the method explanation and negative-result discussion. Keep training inputs, generated evaluation inputs, and retrieval references distinct.
- Text alignment: explain caption supervision, check a small set of existing captions, and interpret the retrieval result.
- Clarify which comparisons use different targets or training settings.

## Conclusion and future work
- Answer the project questions using the results already available.
- Summarize the main limitations briefly.
- Describe possible future comparisons without making them tasks for this report.

## Final pass
- Add a compact appendix with metric definitions, run/checkpoint paths, and reproduction commands.
- Include a brief historical second-prior note if it helps explain the project's progress.
- Verify remaining citations and figure captions against their sources.
- Fix unfinished wording, spelling, inconsistent notation, and duplicated explanations.
- Mark fulfilled TODOs while preserving the original markers.
- Compile and check the final figures, tables, references, and page layout.

Already handled: CoMM direction/path fixes and notebook rerun, separation of its input branches, and documentation of low-level training-budget differences.
