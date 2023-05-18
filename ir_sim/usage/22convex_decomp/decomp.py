from ir_sim.env import EnvBase
import numpy as np

env = EnvBase('grid_map.yaml', control_mode='keyboard', init_args={'no_axis': False}, collision_mode='stop', save_ani=False)
# env = EnvBase('grid_map_car.yaml', control_mode='keyboard', init_args={'no_axis': False}, collision_mode='stop', save_ani=True)

for i in range(3000):
    env.step()
    env.render(show_traj=True, show_polygon=True)
    env.get_lidar_points()
    # calculate the lidar point on the map
    if env.done(): break
        
env.end(ani_name='grid_map', ani_kwargs={'subrectangles': True})
