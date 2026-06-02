#!/bin/bash

seq 1 10 | xargs -I {} -P 5 bash -c ' \
  python examples/custom_example.py --dataset_instance mp3d.17DRP5sb8fy --input_audio /mnt/c/Users/luisvz/Documents/visgraf/soundspaces/data/mp3d_example/sounds/alarm/alarm.wav --sample_rate 44100 --simulation rollout --navigation target --seed "{}" && \
  python examples/audio_processing.py --file "/mnt/c/Users/luisvz/Documents/visgraf/soundspaces/data/mp3d_example/sounds/alarm/simulation/target/seed_{}/output.wav" \
'