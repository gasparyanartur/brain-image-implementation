# Copied from https://github.com/NonaRjb/AlignVis/blob/main/src/dataset/things_eeg_preprocessing.py


import argparse
from pathlib import Path
from brain_image.data.things_eeg2_preprocessing import ThingsEEG2DatasetPreprocessingConfig, mvnn, epoching, save_prepr


def _make_configs_from_args(args: argparse.Namespace) -> ThingsEEG2DatasetPreprocessingConfig:
	"""Instantiate a Configs object from parsed CLI args."""
	    
	return ThingsEEG2DatasetPreprocessingConfig(
		sub=args.sub,
		n_ses=args.n_ses,
		sfreq=args.sfreq,
		mvnn_dim=args.mvnn_dim,
		data_path=args.data_path,
		preprocessed_eeg_dir=args.preprocessed_eeg_dir,
		raw_eeg_dir=args.raw_eeg_dir,
    )


def main(args):
	print('>>> EEG data preprocessing <<<')
	print('\nInput arguments:')
	for key, val in vars(args).items():
		print('{:16} {}'.format(key, val))

	# Set random seed for reproducible results

	configs = _make_configs_from_args(args)

	# =============================================================================
	# Epoch and sort the data
	# =============================================================================
	# Channel selection, epoching, baseline correction and frequency downsampling of
	# the test and training data partitions.
	# Then, the conditions are sorted and the EEG data is reshaped to:
	# Image conditions × EGG repetitions × EEG channels × EEG time points
	# This step is applied independently to the data of each partition and session.
	epoched_test, _, ch_names, times = epoching(configs, 'test', args.seed)
	epoched_train, img_conditions_train, _, _ = epoching(configs, 'training', args.seed)


	# =============================================================================
	# Multivariate Noise Normalization
	# =============================================================================
	# MVNN is applied independently to the data of each session.
	whitened_test, whitened_train = mvnn(configs, epoched_test, epoched_train)
	del epoched_test, epoched_train


	# =============================================================================
	# Merge and save the preprocessed data
	# =============================================================================
	# In this step the data of all sessions is merged into the shape:
	# Image conditions × EGG repetitions × EEG channels × EEG time points
	# Then, the preprocessed data of the test and training data partitions is saved.
	save_prepr(configs, whitened_test, whitened_train, img_conditions_train, ch_names,
		times, args.seed)

# =============================================================================
# Input arguments
# =============================================================================
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--sub', default=10, type=int)
	parser.add_argument('--n_ses', default=4, type=int)
	parser.add_argument('--sfreq', default=250, type=int)
	parser.add_argument('--mvnn_dim', default='epochs', type=str)
	parser.add_argument("-d", "--data_path", default="data/things-eeg2")
	parser.add_argument("--preprocessed_eeg_dir", type=str, default='preprocessed-eeg')
	parser.add_argument("--raw_eeg_dir", type=str, default="raw-eeg")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	main(args)