#!/bin/bash
script_dir="/home/ubuntu/scripts"
script_name="run_scenario.sh"
# Loop through all arguments
# for arg in "$@"; do
#     /run_scenario.sh "$arg"
# done

if [ $# -eq 0 ]; then
    "$script_dir/$script_name" 
else
    for arg in "$@"; do
        "$script_dir/$script_name" "$arg"
    done
fi