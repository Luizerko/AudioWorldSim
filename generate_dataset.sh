#!/bin/bash

seq 1 1000 | xargs -I {} -P 30 bash -c ' \
  python custom_simulator/simulator.py --dataset_instance mp3d.17DRP5sb8fy --input_audio /mnt/f/lz/visgraf/soundspaces/data/mp3d/alarm/alarm.wav --simulation rollout --navigation target --seed "{}" --video && \
  { \
    python custom_simulator/audio_processing.py --file "/mnt/f/lz/visgraf/soundspaces/data/mp3d/alarm/17DRP5sb8fy/rollout/target/seed_{}/output.wav" --method mel & \
    python custom_simulator/audio_processing.py --file "/mnt/f/lz/visgraf/soundspaces/data/mp3d/alarm/17DRP5sb8fy/rollout/target/seed_{}/output.wav" --method stft & \
    wait;
  } \