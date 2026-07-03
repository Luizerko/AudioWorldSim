#!/bin/bash

BASE_DIR="/mnt/f/lz/visgraf/soundspaces/data/mp3d/alarm/17DRP5sb8fy/rollout/target"
OUTPUT_FILE="problematic_seeds.txt"

total_seeds=0
empty_seeds=0
stuck_seeds=0
total_actions=0
> "$OUTPUT_FILE"

# Looping all simulations
for dir in "$BASE_DIR"/seed_*/; do
    seed_name=$(basename "$dir")
    
    # Getting total simulations
    total_seeds=$((total_seeds + 1))

    # Counting broken simulations
    if [ -z "$(ls -A "$dir")" ]; then
        empty_seeds=$((empty_seeds + 1))
        echo "$seed_name - empty" >> "$OUTPUT_FILE"

    # Counting stuck simulations
    elif [ -f "${dir}log_stuck" ]; then
        stuck_seeds=$((stuck_seeds + 1))
        echo "$seed_name - stuck" >> "$OUTPUT_FILE"

    # Processing total time for valid simulations
    else
        action_file="${dir}action_list.txt"
        action_count=$(tr -cd ',' < "$action_file" | wc -c)
        total_actions=$((total_actions + action_count))
    fi
done

echo "Total Navigations:      $total_seeds"
echo "Broken Navigations:     $empty_seeds"
echo "Stuck Navigations:      $stuck_seeds"

total_audio_time=$(awk "BEGIN {print $total_actions*0.2}")
echo "Total Audio Time (s):   $total_audio_time"