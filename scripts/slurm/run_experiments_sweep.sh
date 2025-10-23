experiment_name=$1
array_arg=$2
cli_args="${@:3}"


if [ -z "$experiment_name" ]; then
    echo "Missing experiment_name"
    echo "Usage: $0 experiment_name array_arg args..."
    exit 1
fi
echo "Experiment Name: $experiment_name"


if [ -z "$array_arg" ]; then
    echo "Missing array_arg"
    echo "Usage: $0 experiment_name array_arg args..."
    exit 1
fi  
echo "Array Argument: $array_arg"

echo "CLI Args: $cli_args"

param_dir=${PARAM_DIR:-scripts/slurm/params}
if [ ! -d "$param_dir" ]; then
    echo "Parameter directory does not exist: $param_dir"
    exit 1
fi 
echo "Parameter Directory: $param_dir"

param_path="${param_dir}/${experiment_name}.json"
if [ ! -f "$param_path" ]; then
    echo "Parameter file does not exist: $param_path"
    exit 1
fi 

experiment_dir=${EXPERIMENT_DIR:-experiments}
experiment_path="${experiment_dir}/${experiment_name}"
if [ ! -d "$experiment_path" ]; then
    echo "Experiment directory does not exist: $experiment_path"
    echo "Creating directory: $experiment_path"
    mkdir -p $experiment_path
fi 

echo "Experiment Directory: $experiment_dir"
echo "Parameter Directory: $param_dir"
echo "Experiment Path: $experiment_path"
echo "Parameter Path: $param_path"

sbatch $array_arg scripts/slurm/run_sweep.sh $param_path python /workspace/scripts/train_eeg.py --config-name=train_eeg_alignprior trainer.log_dir=$experiment_path trainer.make_subdir=False $cli_args