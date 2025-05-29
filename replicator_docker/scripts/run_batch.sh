#!/bin/bash
script_dir="/home/ubuntu/scripts"
script_name="run_scenario.sh"
# Loop through all arguments
# for arg in "$@"; do
#     /run_scenario.sh "$arg"
# done

# Time the script
start_time=$(date +%s)

if [ $# -eq 0 ]; then
    "$script_dir/$script_name" 
else
    for arg in "$@"; do
        "$script_dir/$script_name" "$arg"
    done
fi

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "Script completed in $elapsed seconds."