# Copied from https://github.com/NonaRjb/AlignVis/blob/main/src/dataset/things_eeg_preprocessing.py


import argparse
from brain_image.data.things_eeg2_preprocessing import mvnn, epoching, save_prepr


def main(args):
	print('>>> EEG data preprocessing <<<')
	print('\nInput arguments:')
	for key, val in vars(args).items():
		print('{:16} {}'.format(key, val))

	# Set random seed for reproducible results
	seed = 20200220


	# =============================================================================
	# Epoch and sort the data
	# =============================================================================
	# Channel selection, epoching, baseline correction and frequency downsampling of
	# the test and training data partitions.
	# Then, the conditions are sorted and the EEG data is reshaped to:
	# Image conditions × EGG repetitions × EEG channels × EEG time points
	# This step is applied independently to the data of each partition and session.
	epoched_test, _, ch_names, times = epoching(args, 'test', seed)
	epoched_train, img_conditions_train, _, _ = epoching(args, 'training', seed)


	# =============================================================================
	# Multivariate Noise Normalization
	# =============================================================================
	# MVNN is applied independently to the data of each session.
	whitened_test, whitened_train = mvnn(args, epoched_test, epoched_train)
	del epoched_test, epoched_train


	# =============================================================================
	# Merge and save the preprocessed data
	# =============================================================================
	# In this step the data of all sessions is merged into the shape:
	# Image conditions × EGG repetitions × EEG channels × EEG time points
	# Then, the preprocessed data of the test and training data partitions is saved.
	save_prepr(args, whitened_test, whitened_train, img_conditions_train, ch_names,
		times, seed)

# =============================================================================
# Input arguments
# =============================================================================
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--sub', default=10, type=int)
	parser.add_argument('--n_ses', default=4, type=int)
	parser.add_argument('--sfreq', default=250, type=int)
	parser.add_argument('--mvnn_dim', default='epochs', type=str)
	parser.add_argument('--project_dir', default='/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/', type=str)
	args = parser.parse_args()

	main(args)