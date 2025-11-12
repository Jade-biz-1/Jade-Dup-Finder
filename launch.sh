#!/bin/bash
# Launch dupfinder with clean environment
# Run this from a regular terminal, NOT from VS Code's integrated terminal

# Clear all snap-related environment variables
for var in $(env | grep -i snap | cut -d= -f1); do
    unset $var
done

# Set clean library path
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
unset LD_PRELOAD

# Set clean XDG paths
export XDG_DATA_DIRS="/usr/local/share:/usr/share"
export XDG_CONFIG_DIRS="/etc/xdg"

# Run dupfinder with absolute path
cd /home/deepak/Public/dupfinder/build/linux/x64/linux-ninja-cpu
exec ./dupfinder-1.0.0 "$@"
