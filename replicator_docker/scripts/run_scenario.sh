#!/usr/bin/env bash
set -e
set -u

# Check for libGLX_nvidia.so.0 (needed for vulkan)
ldconfig -p | grep libGLX_nvidia.so.0 || NOTFOUND=1
if [[ -v NOTFOUND ]]; then
    cat << EOF > /dev/stderr

Fatal Error: Can't find libGLX_nvidia.so.0...

Ensure running with NVIDIA runtime. (--gpus all) or (--runtime nvidia)

EOF
    exit 1
fi

# Define additional shared libraries for execution
export VK_ICD_FILENAMES=/tmp/nvidia_icd.json
echo "Finding additional dynamic/shared libraries..."
export LD_LIBRARY_PATH=""
for ldpath in $(find -L /opt/nvidia/omniverse/synthetic-data-generation/{exts,extscache} -type d -name bin -or -name deps); do
    echo "  Found: $ldpath"
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$ldpath;
done

# Telemetry consent information
# https://gitlab-master.nvidia.com/omniverse/carbonite/-/blob/master/docs/structuredlog/TelemetryInContainers.rst
echo "Omniverse Software collects installation and configuration details about your software, hardware, and network
configuration (e.g., version of operating system, applications installed, type of hardware, network speed, IP
address) based on our legitimate interest in improving your experience. To improve performance, troubleshooting
and diagnostic purposes of our software, we also collect session behavior, error and crash logs.

Data Collection in container mode is completely anonymous unless specified. 
You may opt-out of this collection anytime by setting the environment variable OMNI_ENV_PRIVACY_CONSENT=0"

echo 'Running: [...]omni.app.synthetic_data_generation.kit' $@


# If no arguments are provided
if [ $# -eq 0 ]; then
    exec "/opt/nvidia/omniverse/synthetic-data-generation/kit/kit" \
    "/opt/nvidia/omniverse/synthetic-data-generation/apps/omni.app.synthetic_data_generation.kit" \
    "--merge-config=/home/ubuntu/telemetry_config.toml" \
    "--/persistent/exts/omni.kit.welcome.window/visible_at_startup=0" \
    "--no-window" \
    "--/omni/replicator/script=/home/ubuntu/scripts/generate_data.py"
else
    for arg in "$@"; do
        echo $arg
        exec "/opt/nvidia/omniverse/synthetic-data-generation/kit/kit" \
        "/opt/nvidia/omniverse/synthetic-data-generation/apps/omni.app.synthetic_data_generation.kit" \
        "--merge-config=/home/ubuntu/telemetry_config.toml" \
        "--/persistent/exts/omni.kit.welcome.window/visible_at_startup=0" \
        "--no-window" \
        "--/omni/replicator/script=/home/ubuntu/scripts/generate_data.py" \
        $arg
    done
fi
# Loop to clean-shutdown telemetry process, shouldn't be needed
while /usr/bin/pgrep omni.telemetry.transmitter >/dev/null; do
    echo "Exiting Omniverse Replicator..."
    sleep 1
done
