#!/bin/bash
SIGNAL_FILE="/home/ubuntu/output/DONE"
LOGFILE="/home/ubuntu/output/LOGS/$(date '+%Y-%m-%d-%H-%M')_data_generation_log.txt"
script_dir="/home/ubuntu/scripts"
script_name="run_scenario.sh"

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
        # Safe to remove — it's empty
        rm "$SIGNAL_FILE"
        echo "Signal file removed."
    fi
fi

# Time the script
start_time=$(date +%s)

# EXECUTE DATA GENERATION CONFIGS
if [ $# -eq 0 ]; then
    "$script_dir/$script_name" 
else
    # Loop through all arguments
    for arg in "$@"; do
        "$script_dir/$script_name" "$arg"
    done
fi

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Script completed in $(format_time $elapsed)."
echo ""
echo ""
echo ""
touch $SIGNAL_FILE