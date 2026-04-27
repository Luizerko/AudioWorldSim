import argparse
import os

import quaternion
import habitat_sim

import numpy as np

from typing import Union

from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.signal import resample

# Get an audio and two sample rates, then resample the first audio based on the second sample rate
def resample_audio(audio1, sample_rate1, sample_rate2):
    if sample_rate1 != sample_rate2:
        audio1 = resample(audio1, int((audio1.shape[0]/sample_rate1)*sample_rate2))
    return audio1

# Pass both the path to the audio to be spatialized and the IR and get the spatialized audio
def convolve_audio(original: str, ir: str, output: Union[str, None] = None, normalize: bool = True):
    sample_rate_dry, dry_audio = wavfile.read(original)
    sample_rate_ir, ir_audio = wavfile.read(ir)

    # Resampling input audio if needed
    dry_audio = resample_audio(dry_audio, sample_rate_dry, sample_rate_ir)

    # Convolving mono or stereo input audio
    if len(dry_audio.shape) == 2:
        dry_audio = np.mean(dry_audio, axis=1)
    spatial_left = fftconvolve(dry_audio, ir_audio[:, 0], mode='full')
    spatial_right = fftconvolve(dry_audio, ir_audio[:, 1], mode='full')

    spatial_audio = np.stack((spatial_left, spatial_right), 1)

    # Normalizing to make it properly audible
    if normalize:
        spatial_audio = spatial_audio / np.max(np.abs(spatial_audio))
        spatial_audio = np.int16(spatial_audio * np.iinfo(np.int16).max)

    # Writing output if aked for
    if output is not None:
        wavfile.write(output, sample_rate_ir, spatial_audio)
    
    return spatial_audio

# Give a list of actions and get back a list of observations, from the initial state until the end state 
def navigation(sim: habitat_sim.Simulator, actions: list[int]):
    observations = [np.array(sim.get_sensor_observations()["audio_sensor"])]
    for action in actions:
        if action == 1:
            observation = np.array(sim.step("move_forward")['audio_sensor'])
        elif action == 2:
            observation = np.array(sim.step("turn_left")['audio_sensor'])
        else:
            observation = np.array(sim.step("turn_right")['audio_sensor'])
        observations.append(observation)
    
    return observations

# Crossfading for mixing audio from adjacent time steps
def crossfade(x1, x2, mixing_time, sample_rate):
    crossfade_samples = int(mixing_time * sample_rate)
    x2_weight = np.arange(crossfade_samples + 1) / crossfade_samples
    x1_weight = np.flip(x2_weight)
    x3 = [x1[:, :crossfade_samples+1] * x1_weight + x2[:, :crossfade_samples+1] * x2_weight, x2[:, crossfade_samples+1:]]

    return np.concatenate(x3, axis=1)

# Get a list of observations through time and get the output audio of the navigation
def convolve_audio_over_time(original: str, observations: list[np.array], sample_rate: int, time_step: float, output: Union[str, None] = None, normalize: bool = True, crossfade_bool: bool = True):
    # Reading input audio, resampling and converting to mono if needed
    sample_rate_dry, dry_audio = wavfile.read(original)
    dry_audio = resample_audio(dry_audio, sample_rate_dry, sample_rate)
    if len(dry_audio.shape) == 2:
        dry_audio = np.mean(dry_audio, axis=1)

    # Navigating and spatializing sound
    num_samples_per_step = int(sample_rate*time_step)
    spatial_audio = []
    last_observation = None
    current_index = 0
    for observation in observations:
        observation = observation.T

        # Computing indices for amount of history needed, start of our audio covolution and end of audio convolution. Notice that we get a 'valid' convolution only when we have an audio segment of length observation.shape[0] + num_samples_per_step - 1
        needed_history = observation.shape[0] - 1
        start_index = current_index - needed_history
        end_index = current_index + num_samples_per_step

        # Handling the case in which we dont have enough sound history yet
        if start_index < 0:
            sound_segment = dry_audio[current_index:end_index]
            spatial_left = fftconvolve(sound_segment, observation[:, 0], mode='full')
            spatial_right = fftconvolve(sound_segment, observation[:, 1], mode='full')
            spatial_left_right = np.stack((spatial_left[:num_samples_per_step], spatial_right[:num_samples_per_step]), 1)    

        # Handling the case in which we have enough history, so we can consider reverb
        else:
            # Handling sound warping in case the file "runs out of sound"    
            if end_index > len(dry_audio):
                sound_segment = np.concatenate((dry_audio[start_index:], dry_audio[:end_index % len(dry_audio)]), axis=0)
            else:
                sound_segment = dry_audio[start_index:end_index]

            spatial_left = fftconvolve(sound_segment, observation[:, 0], mode='valid')
            spatial_right = fftconvolve(sound_segment, observation[:, 1], mode='valid')
            spatial_left_right = np.stack((spatial_left, spatial_right), 1)

        # Crossfading spatial segments if asked for
        if crossfade_bool and last_observation is not None:
            if start_index < 0:
                spatial_left_old = fftconvolve(sound_segment, last_observation[:, 0], mode='full')[:num_samples_per_step]
                spatial_right_old = fftconvolve(sound_segment, last_observation[:, 1], mode='full')[:num_samples_per_step]
            else:
                spatial_left_old = fftconvolve(sound_segment, last_observation[:, 0], mode='valid')
                spatial_right_old = fftconvolve(sound_segment, last_observation[:, 1], mode='valid')

            spatial_left_right_old = np.stack((spatial_left_old, spatial_right_old), axis=1)
            spatial_left_right = crossfade(spatial_left_right_old.T, spatial_left_right.T, 0.05, sample_rate).T

        # Appending current audio spatialization to our spatial audio array
        spatial_audio.append(spatial_left_right)
        
        last_observation = observation
        current_index = (current_index + num_samples_per_step) % len(dry_audio)

    final_audio = np.concatenate(spatial_audio, axis=0)

    # Normalize to make it properly audible
    if normalize:
        final_audio = final_audio / np.max(np.abs(final_audio))
        final_audio = np.int16(final_audio * np.iinfo(np.int16).max)

    if output is not None:
        wavfile.write(output, sample_rate, final_audio)

    return final_audio


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_instance", help="Name of the dot-separate dataset and instance to be used. Example: mp3d.17DRP5sb8fy", type=str, required=True)
    parser.add_argument("--input_audio", help="Path to input audio.", type=str, required=True)
    parser.add_argument("--sample_rate", help="Sample rate for sound simulation and later for audio spatialization.", type=int, default=44100)

    parser.add_argument("--navigation", help="Choose navigation mode. 'static' for single IR computation, 'rollout' for a specified list of actions with their respective observations, and 'interactive' to play around in the audio-based simulation.", type=str, choices=['static', 'rollout', 'interactive'], default='static')

    parser.add_argument("--time_step", help="The amount of time for a step in the kinematic (not dynamic) simulation. Since we don't have physics enabled, the agent teleports. Considering a forward action moves the agent 0.25m, for a reasonable default estimate of time, we use 0.25s per time-step.", type=float, default=0.25)
    
    args = parser.parse_args()

    # Getting dataset and mesh
    dataset, instance = args.dataset_instance.split(".")

    # Scene configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    
    if dataset == "replica":
        backend_cfg.scene_id = f"data/scene_datasets/{dataset}/{instance}/habitat/mesh_semantic.ply"
    elif dataset == "mp3d":
        dataset_aux = dataset + "_example"
        backend_cfg.scene_id = f"data/scene_datasets/{dataset_aux}/{instance}/{instance}.glb"
        
    backend_cfg.scene_dataset_config_file = f"data/scene_datasets/{dataset}/{dataset}.scene_dataset_config.json"
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = False

    # Simulation initialization and agent movement configuration 
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    # agent_cfg.action_space = {
    #     "move_forward": habitat_sim.agent.ActionSpec(
    #         "move_forward", habitat_sim.agent.ActuationSpec(amount=0.2) 
    #     ),
    #     "turn_left": habitat_sim.agent.ActionSpec(
    #         "turn_left", habitat_sim.agent.ActuationSpec(amount=180.0) 
    #     ),
    #     "turn_right": habitat_sim.agent.ActionSpec(
    #         "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
    #     ),
    # }
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # Setting navmesh path for searching navigable points
    if dataset == "replica":
        sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/{dataset}/{instance}/habitat/mesh_semantic.navmesh"))
    elif dataset == "mp3d":
        dataset_aux = dataset + "_example"
        sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/{dataset_aux}/{instance}/{instance}.navmesh"))

    # Acoustics configurations for sensor
    acoustics_cfg = habitat_sim.sensor.RLRAudioPropagationConfiguration()
    acoustics_cfg.sampleRate = args.sample_rate
    acoustics_cfg.indirect = True

    # Channel layout for sensor
    channel_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayout()
    channel_layout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    channel_layout.channelCount = 2

    # Sensor configuration. Avoid setting sensor position so that it automatically follows the agent everywhere
    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    # audio_sensor_spec.position = [0.0, 1.5, 0.0]

    if dataset == "replica":
        audio_sensor_spec.enableMaterials = False
    elif dataset == "mp3d":
        audio_sensor_spec.enableMaterials = False

    audio_sensor_spec.acousticConfig = acoustics_cfg
    audio_sensor_spec.channelLayout = channel_layout
    sim.add_sensor(audio_sensor_spec)

    # Initializing an agent
    agent = sim.initialize_agent(0)
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([1.0, 0.0, 0.0])
    agent_state.rotation = np.array([0.0, 1.0, 0.0, 0.0])
    agent.set_state(agent_state)

    # Initializing a sound source
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(np.array([3.0, 1.5, 0.0]))
    audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")

    # Computing a single IR and spatializing the entire audio based on that response
    if args.navigation == 'static':
        obs = np.array(sim.get_sensor_observations()["audio_sensor"])
        
        static_path = args.input_audio[:args.input_audio.rfind("/")+1] + 'static/'
        os.makedirs(static_path, exist_ok=True)
        ir_path = static_path + 'IR.wav'
        wavfile.write(ir_path, args.sample_rate, obs.T)

        output_path = static_path + 'output.wav'
        convolve_audio(args.input_audio, ir_path, output_path)

    # Simulation rollout for a certain list of actions
    elif args.navigation == 'rollout':
        observations = navigation(sim, [2 for _ in range(36)] + [1 for _ in range(0)])

        # Saving each IR to a file
        navigation_path = args.input_audio[:args.input_audio.rfind("/")+1] + 'navigation/'
        os.makedirs(navigation_path, exist_ok=True)
        ir_paths = [navigation_path + f'IR_{i+1}.wav' for i in range(len(observations))]
        for i, observation in enumerate(observations):
            wavfile.write(ir_paths[i], args.sample_rate, observation.T)

        # import ipdb
        # ipdb.set_trace()

        # Running the simulation on audio
        output_path = navigation_path + 'output.wav'
        convolve_audio_over_time(args.input_audio, observations, args.sample_rate, args.time_step, output_path, True, True)

    # Sanity check for the mesh based on source visibility and ray efficiency
    print(audio_sensor.sourceIsVisible())
    print(audio_sensor.getRayEfficiency())

    sim.close()