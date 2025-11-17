base_log_dir=${BASE_LOG_DIR:-logs/slurm}
mode=$2
logs_id=$1


p=$(find $base_log_dir -name *$logs_id*)
echo $p

if [ $mode == "vim" ]; then
    vim $p
elif [ $mode == "tail" ]; then
    tail -f $p
fi