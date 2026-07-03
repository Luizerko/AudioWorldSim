import sys
import quaternion
import magnum 
import habitat_sim
from habitat_sim.utils.datasets_download import main

if __name__ == "__main__":
    args = [
        "--uids", "mp3d_example_scene", 
        "--data-path", "data/"
    ]
    main(args)