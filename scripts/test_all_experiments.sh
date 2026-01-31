test_script=$1
echo "Test Script: $test_script"

experiment_dir=$2
echo "Experiment Directory: $experiment_dir"

experiment_files=$(ls $experiment_dir)
echo "Experiment Files:"
echo $experiment_files

for f in $experiment_files; do
    echo "Testing $f..."
    python $test_script $experiment_dir/$f
done