# MRI-QRF execution order

1. `organize_oasis1.py`
2. `organize_oasis2.py`
3. `split_subjects.py`
4. `preprocess_to_128.py`
5. `generate_distortions.py`
6. `derive_resolutions.py` for 64 and 32 clean/distorted copies
7. `validate_prepared_data.py`
8. Classical ML:
   - run `run_ml_models_mri_qrf.py` at 32
   - run at 64
   - run at 128
9. `subject_level_analysis_mri_qrf.py`
10. DL:
   - `validate_dl_data.sh`
   - `run_dl_all.sh` or `start_dl_tmux.sh`
11. Statistical preparation:
   - `prepare_glmm_input.py`
   - `mixed_effects_logistic.R`
   - `patient_cluster_bootstrap.py`
12. Reader study:
   - `prepare_rater_study.py`
   - deploy Apps Script interface
13. Upload trained models/results into the prepared repository directories
14. `generate_checksums.py`
15. Create a tagged GitHub release
