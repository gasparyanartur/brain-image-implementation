experiment_dir=$1
echo "Experiment Directory: $experiment_dir"


experiment_files=$(ls $experiment_dir)
echo "Experiment Files:"
echo $experiment_files

for f in $experiment_files; do
    echo "Testing $f..."
    python scripts/test_eeg.py $experiment_dir/$f
done