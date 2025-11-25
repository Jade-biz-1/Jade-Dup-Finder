#!/bin/bash
# Wrapper script to run dupfinder with correct library paths

# Remove any snap library paths
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v snap | tr '\n' ':')

# Add system library paths
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# Unset any preload
unset LD_PRELOAD

# Run dupfinder from build directory
cd "$(dirname "$0")/build/linux/x64/linux-ninja-cpu"
exec ./dupfinder "$@"
