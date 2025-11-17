
#!/bin/bash

#GLOBALS
SIGNAL_FILE="/home/csrobot/Omniverse/SynthData/DONE"
SKIP_WAIT=true 
EXPERIMENT_NAME="atb1_SCRIPTTEST"

LOGFILE="/home/csrobot/synth_perception/experiment_log.txt"
CONVERT_SCRIPT="/home/csrobot/synth_perception/scripts/replicator_to_yolo_dataset.py"
TRAIN_SCRIPT="/home/csrobot/synth_perception/scripts/yolo_train.py"
BENCHMARK_SCRIPT="/home/csrobot/synth_perception/scripts/benchmark_yolo_model.py"

#APPEND ALL OUTPUT TO LOGFILE
exec > >(tee -a $LOGFILE) 2>&1

# SETUP FUNCTIONS
format_time() {
    local s=$1
    printf "%02d:%02d:%02d" $((s/3600)) $((s%3600/60)) $((s%60))
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

run_step() {
    local label="$1"
    echo ""
    echo "[$(timestamp)] Starting: $label"

    local start=$(date +%s.%N)

    # Run the command passed as the rest of the arguments
    shift
    "$@"

    local end=$(date +%s.%N)
    local duration=$(printf "%.1f" "$(echo "$end - $start" | bc)")

    echo "[$(timestamp)] Finished: $label (took ${duration}s)"
    echo
}

# START EXECUTION
echo "============================================================="
echo "[$(timestamp)] STARTING EXPERIMENT: $EXPERIMENT_NAME"
# WAIT FOR SIGNAL FILE SAYING DATA GENERATION IS FINISHED
SECONDS=0
if [ "$SKIP_WAIT" = false ]; then
    echo "Waiting for done signal from data generation..."
    while [ ! -f $SIGNAL_FILE ]; do
        sleep 1
    done
else
    echo "Skipping wait for signal file."
fi
echo "[$(timestamp)] DATA GENERATION FINISHED (SIGNAL FOUND)"
elapsed=$(format_time $SECONDS)
echo "[$(timestamp)] Waited for $elapsed"

# CALL EXPERIMENTAL PROCEDURE SCRIPTS
run_step "Convert Dataset" python3 $CONVERT_SCRIPT $EXPERIMENT_NAME dev/exp
run_step "Train YOLO Model" python3 $TRAIN_SCRIPT --name $EXPERIMENT_NAME --epochs 10
run_step "Run Benchmark v0" python3 $BENCHMARK_SCRIPT --model $EXPERIMENT_NAME

# CLEAN UP SIGNAL FILE
if [ -f "$SIGNAL_FILE" ]; then
    if [ -s "$SIGNAL_FILE" ]; then
        echo "WARNING: Signal file '$SIGNAL_FILE' is NOT empty. Leaving it untouched."
    else
        # Safe to remove — it's empty
        rm "$SIGNAL_FILE"
        echo "Signal file removed."
    fi
fi