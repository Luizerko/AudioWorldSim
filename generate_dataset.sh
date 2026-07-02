#!/bin/bash

seq 1 1000 | xargs -I {} -P 30 bash -c ' \
  python examples/custom_simulator.py --dataset_instance mp3d.17DRP5sb8fy --input_audio /mnt/c/Users/luisvz/Documents/visgraf/soundspaces/data/mp3d_example/sounds/alarm/alarm.wav --sample_rate 44100 --simulation rollout --navigation target --seed "{}" --video && \
  { \
    python examples/audio_processing.py --file "/mnt/c/Users/luisvz/Documents/visgraf/soundspaces/data/mp3d_example/sounds/alarm/simulation/target/seed_{}/output.wav" --method mel & \
    python examples/audio_processing.py --file "/mnt/c/Users/luisvz/Documents/visgraf/soundspaces/data/mp3d_example/sounds/alarm/simulation/target/seed_{}/output.wav" --method stft & \
    wait;
  } \