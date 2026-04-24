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
    # if len(dry_audio.shape) == 1:
    #     spatial_left = fftconvolve(dry_audio, ir_audio[:, 0], mode='full')
    #     spatial_right = fftconvolve(dry_audio, ir_audio[:, 1], mode='full')
    # elif dry_audio.shape[1] == 2:
    #     spatial_left = fftconvolve(dry_audio[:, 0], ir_audio[:, 0], mode='full')
    #     spatial_right = fftconvolve(dry_audio[:, 1], ir_audio[:, 1], mode='full')
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
def navigation(sim, actions: list[int]):
    observations = [np.array(sim.get_sensor_observations()["audio_sensor"])]
    max_len_obs = 0
    for action in actions:
        if action == 1:
            observation = np.array(sim.step("move_forward")['audio_sensor'])
        elif action == 2:
            observation = np.array(sim.step("turn_left")['audio_sensor'])
        else:
            observation = np.array(sim.step("turn_right")['audio_sensor'])
        observations.append(observation)
        
        if observation.shape[1] > max_len_obs:
            max_len_obs = observation.shape[1]

    # Padding observations eith the maximum obsevation length because different poses in space can generate IRs of different length
    padded_observations = [np.pad(observation, pad_width=((0, 0), (0, max_len_obs - observation.shape[1])), mode='constant') for observation in observations]
    
    return np.stack(padded_observations, 0)

### NEEDS TO BE REIMPLEMENTED IF USED ONE DAY ###
# Convolving IRs over time on audio to simulate spatial navigation
# def convolve_audio_over_time(original: str, irs: list[str], sample_rate_ir: int, time_step: float = 0.25, output: Union[str, None] = None):
#     # Splitting original audio into segments of time_step seconds
#     sample_rate_dry, dry_audio = wavfile.read(original)
#     dry_audio = resample_audio(dry_audio, sample_rate_dry, sample_rate_ir)
    
#     split_size = int(sample_rate_ir*time_step)
#     dry_audios = np.split(dry_audio, [split_size*(i+1) for i in range(len(irs))], axis=0)[:-1]

#     navigation_path = original[:original.rfind("/")+1] + 'navigation/'
#     original_paths = [navigation_path + f'original_{i+1}.wav' for i in range(len(dry_audios))]
#     for i, dry_audio in enumerate(dry_audios):
#         wavfile.write(original_paths[i], sample_rate_ir, dry_audio)

#     # Spatializing every piece of audio
#     spatialized_audios = [convolve_audio(original, ir, normalize=False) for original, ir in zip(original_paths, irs)]

#     # Combining spatialized audio into single output
#     final_audio_size = split_size * (len(irs) - 1) + len(spatialized_audios[-1])
#     spatialized_navigation = np.zeros((final_audio_size, 2))
#     for i, spatialized_audio in enumerate(spatialized_audios):
#         spatialized_navigation[i*split_size : i*split_size + len(spatialized_audio)] += spatialized_audio

#     # Normalizing final audio
#     spatialized_navigation = spatialized_navigation / np.max(np.abs(spatialized_navigation))
#     spatialized_navigation = np.int16(spatialized_navigation * np.iinfo(np.int16).max)

#     if output is not None:
#         wavfile.write(output, sample_rate_ir, spatialized_navigation)

#     return spatialized_navigation

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--replica_mesh", help="name of replica mesh to be used.", type=str, required=True)
    parser.add_argument("--input_audio", help="Path to input audio.", type=str, required=True)
    parser.add_argument("--sample_rate", help="Sample rate for sound simulation and later for audio spatialization.", type=int, default=44100)

    parser.add_argument("--navigation", help="Choose navigation mode. 'static' for single IR computation, 'rollout' for a specified list of actions with their respective observations, and 'interactive' to play around in the audio-based simulation.", type=str, choices=['static', 'rollout', 'interactive'], default='static')

    parser.add_argument("--time_step", help="The amount of time for a step in the kinematic (not dynamic) simulation. Since we don't have physics enabled, the agent teleports. Considering a forward action moves the agent 0.25m, for a reasonable default estimate of time, we use 0.25s per time-step.", type=float, default=0.25)
    
    args = parser.parse_args()

    # Scene configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = f"data/scene_datasets/replica/{args.replica_mesh}/habitat/mesh_semantic.ply"
    backend_cfg.scene_dataset_config_file = "data/scene_datasets/replica/replica.scene_dataset_config.json"
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
    sim.pathfinder.load_nav_mesh(os.path.join(f"data/scene_datasets/replica/{args.replica_mesh}/habitat/mesh_semantic.navmesh"))

    # Acoustics configurations for sensor
    acoustics_cfg = habitat_sim.sensor.RLRAudioPropagationConfiguration()
    acoustics_cfg.sampleRate = args.sample_rate
    acoustics_cfg.indirect = True

    # Channel layout for sensor
    channel_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayout()
    channel_layout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
    channel_layout.channelCount = 2

    # Sensor configuration
    audio_sensor_spec = habitat_sim.AudioSensorSpec()
    audio_sensor_spec.uuid = "audio_sensor"
    audio_sensor_spec.position = [0.0, 1.5, 0.0]
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
    audio_sensor.setAudioSourceTransform(np.array([1.0, 1.5, 0.0]))
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
        observations = navigation(sim, [2 for _ in range(1)] + [1 for _ in range(0)])

        # Saving each IR to a file
        navigation_path = args.input_audio[:args.input_audio.rfind("/")+1] + 'navigation/'
        os.makedirs(navigation_path, exist_ok=True)
        ir_paths = [navigation_path + f'IR_{i+1}.wav' for i in range(observations.shape[0])]
        for i, observation in enumerate(observations):
            wavfile.write(ir_paths[i], args.sample_rate, observation.T)

        # import ipdb
        # ipdb.set_trace()

        # Running the simulation on audio
        # output_path = navigation_path + 'output.wav'
        # convolve_audio_over_time(args.input_audio, ir_paths, args.sample_rate, args.time_step, output_path)

    # Sanity check for the mesh based on source visibility and ray efficiency
    print(audio_sensor.sourceIsVisible())
    print(audio_sensor.getRayEfficiency())

    sim.close()