import argparse
import math
import random
import os

import quaternion
import habitat_sim

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
def simple_navigation(sim: habitat_sim.Simulator, actions: list[int]):
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
    x2_weight = np.arange(crossfade_samples+1) / crossfade_samples
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
    warped = False

    # Zero padding the tail of the IRs so that we don't have time-shift misalingment between audios later on, when we crossfade current IR and past IR (both with current audio segment)
    max_ir_length = max([observation.shape[1] for observation in observations])
    padded_observations = []
    for observation in observations:
        pad_length = max_ir_length - observation.shape[1]
        if pad_length > 0:
            observation = np.pad(observation, ((0, 0), (0, pad_length)), mode='constant')
        padded_observations.append(observation)
    
    for observation in padded_observations:
        observation = observation.T

        # Computing indices for amount of history needed, start of our audio covolution and end of audio convolution. Notice that we get a 'valid' convolution only when we have an audio segment of length observation.shape[0] + num_samples_per_step - 1
        needed_history = observation.shape[0] - 1
        start_index = current_index - needed_history
        end_index = current_index + num_samples_per_step

        # Handling the case in which we dont have enough sound history yet or we have finished the audio file
        if start_index < 0:
            # Handling sound warping in case the file "runs out of sound"
            if warped:
                sound_segment = np.concatenate((dry_audio[start_index:], dry_audio[:end_index]), axis=0)
            else:
                silence = np.zeros(abs(start_index))
                actual_sound = dry_audio[:end_index]
                sound_segment = np.concatenate((silence, actual_sound))

        # Handling the case in which we have enough history, so we can consider reverb
        else:
            # Handling sound warping in case the file "runs out of sound"    
            if end_index > len(dry_audio):
                sound_segment = np.concatenate((dry_audio[start_index:], dry_audio[:end_index % len(dry_audio)]), axis=0)
                warped = True
            else:
                sound_segment = dry_audio[start_index:end_index]

        spatial_left = fftconvolve(sound_segment, observation[:, 0], mode='valid')
        spatial_right = fftconvolve(sound_segment, observation[:, 1], mode='valid')
        spatial_left_right = np.stack((spatial_left, spatial_right), 1)

        # Crossfading spatial segments if asked for. We crossfade the current sound segment + current IR and current sound segment + past IR to simulate how the current sound would have sounded from the previous position
        if crossfade_bool and last_observation is not None:
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


# Plotting navmesh
def visualize_navmesh(sim: habitat_sim.Simulator, filename: str = None, meters_per_pixel: float = 0.01, height: float = 0.0, trajectory=[], inter_pos=[], inter_direcs=[], video: bool = False):
    # Plotting topdown map
    topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
    view_image = np.uint8(topdown_map * 255)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(view_image, cmap="gray")
    ax.set_title("Navigable Area (+ Agent Navigation)")
    ax.axis("off")

    # Helper function to capture current state of the plot for video
    frames = []
    def capture_frame():
        elements = []
        for img in ax.images:
            elements.append(img)
        for line in ax.lines:
            elements.append(line)
        for collection in ax.collections:
            elements.append(collection)
        frames.append(elements)

    # Plotting goal trajectory of the agent
    bounds = sim.pathfinder.get_bounds()
    for i, point in enumerate(trajectory):
        px = (point[0] - bounds[0][0])/meters_per_pixel
        py = (point[2] - bounds[0][2])/meters_per_pixel
        
        # Plotting start point, end point, and all other points in between.
        if i == 0:
            plt.plot(px, py, marker="o", markersize=6, alpha=0.8, color='b')
        elif i == len(trajectory)-1:
            plt.plot(px, py, marker="o", markersize=6, alpha=0.8, color='r')
        else:
            plt.plot(px, py, marker="o", markersize=6, alpha=0.8, color='g')

    # Plotting step by step trajetory of the agent
    bounds = sim.pathfinder.get_bounds()
    for i, point in enumerate(trajectory):
        px = (point[0] - bounds[0][0])/meters_per_pixel
        py = (point[2] - bounds[0][2])/meters_per_pixel

        # We also plot the intermediate points that our agent passes through and intermediate directions that our agent looks towards
        if len(inter_direcs) > 0 and i < len(trajectory)-1:
            for v in inter_direcs[i]:
                plt.quiver(px, py, v[0], -v[2], scale=35, units='width', width=0.003, alpha=0.6, color='g')
                
                if video:
                    capture_frame()

        if len(inter_pos) > 0 and i < len(trajectory)-1:
            for p in inter_pos[i]:
                px = (p[0] - bounds[0][0])/meters_per_pixel
                py = (p[2] - bounds[0][2])/meters_per_pixel
                plt.plot(px, py, marker="x", markersize=2, alpha=0.6, color='g')

                if video:
                    capture_frame()

    # Plotting or saving
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches="tight", dpi=300)

        # Creating video
        if video:
            ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True, repeat_delay=1000)
            ani.save(filename + '.mp4', writer='ffmpeg', dpi=300)
    plt.close()


# Uses the navmesh to sample start and end points, then uses the the habitat sim shortest path to make an action plan for movement and finally moves the agent with simulation actions so it can get from start to finish. Audio sensor observations are captured throughout the whole process
def target_navigation(sim: habitat_sim.Simulator, agent: habitat_sim.Agent, sensor: habitat_sim.AudioSensor, seed: int = 3, filename: str = None, video: bool = False):
    # Getting a random (but valid) start and end
    sim.pathfinder.seed(seed)
    sample1 = sim.pathfinder.get_random_navigable_point()
    sample2 = sim.pathfinder.get_random_navigable_point()

    # Computing shortest path (if any) from start to end
    path = habitat_sim.ShortestPath()
    path.requested_start = sample1
    path.requested_end = sample2
    found_path = sim.pathfinder.find_path(path)
    # geodesic_distance = path.geodesic_distance
    path_points = path.points

    # Returning no observations if no path was found
    if not found_path:
        return []

    # Putting sound at the target, but at the agent's height
    target_position = np.copy(path_points[-1])
    target_position[1] = 1.5
    sensor.setAudioSourceTransform(target_position)

    # Initializing agent on initial position and no rotation (front to [0, 0, -1])
    agent_state = habitat_sim.AgentState()
    agent_state.position = path_points[0]
    agent_state.rotation = np.array([0.0, 0.0, 0.0, 1.0])
    agent.set_state(agent_state)

    # Making agent go through trajectory while saving audio observations and overall steps for plotting later on
    observations = []
    inter_pos = []
    inter_direcs = []
    for point in path_points[1:]:
        inter_pos_aux = []
        inter_direcs_aux = [habitat_sim.utils.common.quat_rotate_vector(agent.get_state().rotation, np.array([0.0, 0.0, -1.0]))]
        while True:
            # Computing current forward of the agent
            current_state = agent.get_state()
            heading_vector = habitat_sim.utils.common.quat_rotate_vector(current_state.rotation, np.array([0.0, 0.0, -1.0]))
            heading_angle = math.atan2(heading_vector[0], -heading_vector[2])
            
            # Computing the vector that points from current poistion to next position
            dx = point[0] - current_state.position[0]
            dz = point[2] - current_state.position[2]
            target_angle = math.atan2(dx, -dz)
        
            # Making agent turn until it's aligned enough with the next point
            angle_diff = (target_angle - heading_angle + math.pi) % (2*math.pi) - math.pi
            if abs(angle_diff) > math.radians(1.6):
                if angle_diff > 0:
                    observations.append(np.array(sim.step("turn_right")['audio_sensor']))
                else:
                    observations.append(np.array(sim.step("turn_left")['audio_sensor']))
            else:
                inter_direcs_aux.append(heading_vector)
                break

        prev_dist = np.inf
        while True:
            # Computing target vector
            current_state = agent.get_state()
            inter_pos_aux.append(current_state.position)
            target_vector = np.array([point[0] - current_state.position[0], point[2] - current_state.position[2]])
            
            # Taking steps forward until agent reaches the (next) target point
            target_dist = np.linalg.norm(target_vector)
            if target_dist > 0.085:
                # Making sure the agent doesn't get stuck
                if prev_dist <= target_dist:
                    agent_state = habitat_sim.AgentState()
                    agent_state.position = point
                    agent_state.rotation = current_state.rotation
                    agent.set_state(agent_state)

                    observations.append(np.array(sim.get_sensor_observations()["audio_sensor"]))
                    break

                prev_dist = target_dist
                observations.append(np.array(sim.step("move_forward")['audio_sensor']))
            else:
                agent_state = habitat_sim.AgentState()
                agent_state.position = point
                agent_state.rotation = current_state.rotation
                agent.set_state(agent_state)
                
                observations.append(np.array(sim.get_sensor_observations()["audio_sensor"]))
                break

        inter_pos.append(inter_pos_aux)
        inter_direcs.append(inter_direcs_aux)


    visualize_navmesh(sim, filename, trajectory=path_points, inter_pos=inter_pos, inter_direcs=inter_direcs, video=video)

    return observations


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_instance", help="Name of the dot-separate dataset and instance to be used. Example: mp3d.17DRP5sb8fy", type=str, required=True)
    parser.add_argument("--input_audio", help="Path to input audio.", type=str, required=True)
    parser.add_argument("--sample_rate", help="Sample rate for sound simulation and later for audio spatialization.", type=int, default=44100)

    parser.add_argument("--simulation", help="Choose simulation mode. 'static' for single IR computation, 'rollout' for a specified list of actions with their respective observations, and 'interactive' to play around in the audio-based simulation.", type=str, choices=['static', 'rollout', 'interactive'], default='static')
    parser.add_argument("--navigation", help="Choose navigation mode (for rollout simulation only). If rollout simulation was chosen, choose how your agent will navigate the simulation. If 'simple', agent will take some unverified rotations and move forward. If 'target', agent will use the navmesh to try and navigate from point A to point B.", type=str, choices=['simple', 'target'], default='simple')

    parser.add_argument("--time_step", help="The amount of time for a step in the kinematic (not dynamic) simulation. Since we don't have physics enabled, the agent teleports. Considering a forward action moves the agent 0.15m, for a reasonable default estimate of time, we use 0.2s per time-step.", type=float, default=0.2)

    parser.add_argument("--verbose", help="Print and plot everything. Meant for debugging.", action='store_true')
    parser.add_argument("--video", help="Create a video out of the Navmesh plots. Meant for better visualization.", action='store_true')
    parser.add_argument("--seed", help="Choose a specific seed for the target navigation scenario.", type=int)
    
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
    
    # Configuring amount of movement per step
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.15) 
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=3.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=3.0)
        ),
    }

    # Image sensor configuration. This part of the code won't work until I reinstall Habitat-sim with proper configuration for imagery
    # rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    # rgb_sensor_spec.uuid = "color_sensor"
    # rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    # height, width = 256, 256
    # rgb_sensor_spec.resolution = [height, width]
    # agent_cfg.sensor_specifications = [rgb_sensor_spec]

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

    # Audio sensor configuration. Following SoundSpaces' continuous_simulator.py, we avoid setting sensor position so that it automatically follows the agent everywhere
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
    # agent_state.rotation = np.array([0.0, 1.0, 0.0, 0.0])
    agent.set_state(agent_state)

    # Initializing a sound source
    audio_sensor = sim.get_agent(0)._sensors["audio_sensor"]
    audio_sensor.setAudioSourceTransform(np.array([3.0, 1.5, 0.0]))
    audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")

    # Visualizing navmesh for sanity check
    if args.verbose:
        height = sim.pathfinder.get_random_navigable_point()[1]
        visualize_navmesh(sim, meters_per_pixel=0.01, height=height)

    # Computing a single IR and spatializing the entire audio based on that response
    if args.simulation == 'static':
        obs = np.array(sim.get_sensor_observations()["audio_sensor"])
        
        static_path = args.input_audio[:args.input_audio.rfind("/")+1] + 'static/'
        os.makedirs(static_path, exist_ok=True)
        ir_path = static_path + 'IR.wav'
        wavfile.write(ir_path, args.sample_rate, obs.T)

        output_path = static_path + 'output.wav'
        convolve_audio(args.input_audio, ir_path, output_path)

    # Simulation rollout for a certain list of actions
    elif args.simulation == 'rollout':
        # Agent navigation for simple actions we choose beforehand
        if args.navigation == 'simple':
            # 360 degrees rotation
            observations = simple_navigation(sim, [2 for _ in range(72)])

            # 180 degrees rotation
            # observations = simple_navigation(sim, [2 for _ in range(36)])
            
            # 90 degrees rotation + forward movement
            # observations = simple_navigation(sim, [2 for _ in range(18)] + [1 for _ in range(10)])

            # Saving each IR to a file
            navigation_path = args.input_audio[:args.input_audio.rfind("/")+1] + f'simulation/{args.navigation}/'
            os.makedirs(navigation_path, exist_ok=True)
            ir_paths = [navigation_path + f'IR_{i+1}.wav' for i in range(len(observations))]
            for i, observation in enumerate(observations):
                wavfile.write(ir_paths[i], args.sample_rate, observation.T)

        # Agent navigation for random start and end points, with agent moving in the shortest path possible between them
        elif args.navigation == 'target':
            # Tested seeds:
            # 3, 776, 249, 573 -> Easy seeds
            # 487, 275, 779 -> Medium seeds
            # 420, 378 -> Hard seeds
            if args.seed is None:
                seed = random.randint(0, 1000)
            else:
                seed = args.seed
            print(seed)

            # Creating seed folder, but not saving each IR
            navigation_path = args.input_audio[:args.input_audio.rfind("/")+1] + f'simulation/{args.navigation}/seed_{seed}/'
            os.makedirs(navigation_path, exist_ok=True)

            observations = target_navigation(sim, agent, audio_sensor, seed, navigation_path+"navigation", args.video)

        # Running the simulation on audio and saving it
        output_path = navigation_path + 'output.wav'
        convolve_audio_over_time(args.input_audio, observations, args.sample_rate, args.time_step, output_path, True, True)

    # Sanity check for the mesh based on source visibility and ray efficiency
    if args.verbose:
        print(audio_sensor.sourceIsVisible())
        print(audio_sensor.getRayEfficiency())

    sim.close()