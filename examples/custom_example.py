import argparse
import quaternion
import habitat_sim
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy.signal import resample

# import ipdb
# ipdb.set_trace()

# Pass both the path to the audio to be spatialized and the IR and get the spatialized audio
def convolve_audio(original: str, ir:str, output: str):
    sample_rate_dry, dry_audio = wavfile.read(original)
    sample_rate_ir, ir_audio = wavfile.read(ir)

    # Resampling input audio if needed
    if sample_rate_dry != sample_rate_ir:
        dry_audio = resample(dry_audio, int((dry_audio.shape[0]/sample_rate_dry)*sample_rate_ir))

    # Convolving mono or stereo input audio
    if len(dry_audio.shape) == 1:
        spatial_left = fftconvolve(dry_audio, ir_audio[:, 0], mode='full')
        spatial_right = fftconvolve(dry_audio, ir_audio[:, 1], mode='full')
    elif dry_audio.shape[1] == 2:
        spatial_left = fftconvolve(dry_audio[:, 0], ir_audio[:, 0], mode='full')
        spatial_right = fftconvolve(dry_audio[:, 1], ir_audio[:, 1], mode='full')
    spatial_audio = np.stack((spatial_left, spatial_right), 1)

    # Normalizing to make it properly audible
    spatial_audio = spatial_audio / np.max(np.abs(spatial_audio))
    spatial_audio_int16 = np.int16(spatial_audio * np.iinfo(np.int16).max)

    wavfile.write(output, sample_rate_ir, spatial_audio_int16)

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

# Convolving IRs over time on audio to simulate spatial navigation
def convolve_audio_over_time(original: str):
    
    return

if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_audio", help="Path to input audio.", type=str, required=True)
    
    parser.add_argument("--navigation", help="Choose navigation mode. 'static' for single IR computation, 'rollout' for a specified list of actions with their respective observations, and 'interactive' to play around in the audio-based simulation.", type=str, choices=['static', 'rollout', 'interactive'], default='static')
    
    parser.add_argument("--sample_rate", help="Sample rate for sound simulation and later for audio spatialization.", type=int, default=44100)
    
    args = parser.parse_args()

    # Scene configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = "data/scene_datasets/replica/apartment_0/habitat/mesh_semantic.ply"
    backend_cfg.scene_dataset_config_file = "data/scene_datasets/replica/replica.scene_dataset_config.json"
    backend_cfg.load_semantic_mesh = True
    backend_cfg.enable_physics = False

    # Simulation initialization 
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

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
    agent_state.position = np.array([0.5, 0.8, 0])
    agent.set_state(agent_state)

    # Initializing a sound source
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(np.array([2.0, 1.5, 0.50]))
    audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")

    # Computing a single IR and spatializing the entire audio based on that response
    if args.navigation == 'static':
        obs = np.array(sim.get_sensor_observations()["audio_sensor"])
        ir_path = args.input_audio[:args.input_audio.rfind("/")+1] + 'IR.wav'
        wavfile.write(ir_path, args.sample_rate, obs.T)

        output_path = args.input_audio[:args.input_audio.rfind("/")+1] + 'output.wav'
        convolve_audio(args.input_audio, ir_path, output_path)

    # Simulation rollout for a certain list of actions
    elif args.navigation == 'rollout':
        observations = navigation(sim, [2 for _ in range(5)] + [1 for _ in range(5)])



        # We don't have physics enabled, so the simulation runs on kinematic mode, meaning agent "teleports". Considering a forward action moves the agent 0.25m, for a reasonable estimate of time, we use 0.25s per time-step


    sim.close()