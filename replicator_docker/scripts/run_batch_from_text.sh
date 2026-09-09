#!/bin/bash
SIGNAL_FILE="/home/ubuntu/output/DONE"
LOGFILE="/home/ubuntu/output/LOGS/$(date '+%Y-%m-%d-%H-%M')_data_generation_log.txt"
script_dir="/home/ubuntu/scripts"
script_name="run_scenario.sh"

DEFAULT_BATCHFILE="/home/ubuntu/config/config_batch.txt"

format_time() {
    local s=$1
    printf "%02d:%02d:%02d" $((s/3600)) $((s%3600/60)) $((s%60))
}

#SEND ALL OUTPUT TO LOGFILE (OVERWRITE)
exec > >(tee $LOGFILE) 2>&1

# CLEAN UP SIGNAL FILE IF IT EXISTS
echo $(date "+%Y-%m-%d %H:%M:%S")
if [ -f "$SIGNAL_FILE" ]; then
    if [ -s "$SIGNAL_FILE" ]; then
        echo "WARNING: Signal file '$SIGNAL_FILE' is NOT empty. Leaving it untouched."
    else
        rm "$SIGNAL_FILE"
        echo "Signal file removed."
    fi
fi

# Time the script
start_time=$(date +%s)

# EXECUTE DATA GENERATION CONFIGS FROM TEXT FILE
# if [ $# -eq 0 ]; then
#     echo "Usage: $0 /path/to/args.txt"
#     exit 1
# fi

input_file="${1:-$DEFAULT_BATCHFILE}"

if [ ! -f "$input_file" ]; then
    echo "ERROR: Input file '$input_file' not found."
    exit 1
fi

while IFS= read -r line || [ -n "$line" ]; do
    # Skip empty lines and comments
    [[ -z "$line" || "$line" == \#* ]] && continue
    echo "Running with arg: $line"
    "$script_dir/$script_name" "$line"
done < "$input_file"

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Script completed in $(format_time $elapsed)."
echo ""
echo ""
echo ""
touch $SIGNAL_FILE
