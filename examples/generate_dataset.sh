#!/bin/bash

for i in {1..10}
do
    python examples/custom_example.py --dataset_instance mp3d.17DRP5sb8fy --input_audio /mnt/c/Users/luisvz/Documents/visgraf/soundspaces/data/mp3d_example/sounds/alarm/alarm.wav --sample_rate 44100 --simulation rollout --navigation target --seed $i &

    python examples/audio_processing.py --file /mnt/c/Users/luisvz/Documents/visgraf/soundspaces/data/mp3d_example/sounds/alarm/simulation/target/seed_$i/output.wav &
done