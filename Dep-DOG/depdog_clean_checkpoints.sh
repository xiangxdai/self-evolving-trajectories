#!/bin/bash

# ================= Configuration =================
# Target extension (this should be .pt based on your description; change it here if you need .pkl)
EXT="*.pt"

# Target file to delete
DELETE_TARGET="ckpt_latest.pt"

# Mode: "dry_run" (print only, do not delete) or "execute" (perform deletion)
# It is safer to run dry_run first to verify the matches
#MODE="dry_run" 
MODE="execute" 
# ===========================================

echo "Scanning the current directory and its subdirectories..."
echo "Mode: $MODE"
echo "Search condition: the directory contains exactly 2 $EXT files"
echo "-------------------------------------------"

# Scan all directories
find . -type d | while read -r dir; do
    # Count how many .pt files exist directly under this directory (excluding subdirectories)
    count=$(find "$dir" -maxdepth 1 -type f -name "$EXT" | wc -l)

    # Apply the rule only when the count is exactly 2
    if [ "$count" -eq 2 ]; then
        
        # Check whether this directory contains the target file to delete
        if [ -f "$dir/$DELETE_TARGET" ]; then
            
            if [ "$MODE" == "execute" ]; then
                rm "$dir/$DELETE_TARGET"
                echo "[deleted] $dir/$DELETE_TARGET"
            else
                echo "[would delete] $dir/$DELETE_TARGET (this directory contains 2 pt files)"
            fi
        fi
    fi
done

echo "-------------------------------------------"
if [ "$MODE" == "dry_run" ]; then
    echo "Dry run completed. Change MODE in the script to 'execute' to perform the actual deletion."
else
    echo "Cleanup complete."
fi