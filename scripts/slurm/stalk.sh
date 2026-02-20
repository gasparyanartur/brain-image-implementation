base_log_dir=${BASE_LOG_DIR:-logs/slurm}
mode=$1
logs_id=$2

# Wait for the log file(s) to appear
p=""
while [ -z "$p" ]; do
    p=$(find $base_log_dir -name "*$logs_id*" 2>/dev/null)
    [ -z "$p" ] && sleep 1
done

echo $p

if [ $mode = "vim" ]; then
    vim $p
elif [ $mode = "tail" ]; then
    tail -f $p
fi