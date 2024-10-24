"""
Floating offshore wind farm yaw control via model-based deep reinforcement learning
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
from floris import FlorisModel
import floris.flow_visualization as flowviz
from mooring_matrix import moor_matrix
from py_wake.examples.data import example_data_path
import floris.layout_visualization as layoutviz
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap

config = {
    "font.family": 'Times New Roman',
    "axes.unicode_minus": False,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
}
rcParams.update(config)


class Environment:
    def __init__(self, evaluation):
        """
         collecting data files
        """
        # simulation setup
        self.fmodel = FlorisModel(r"..\inputs_floating\emgauss_floating.yaml")

        self.iteration_num = 6
        x, y = np.meshgrid(np.linspace(0, 4 * 1600, 5), np.linspace(0, 3 * 1600, 4))
        self.x = x.flatten()
        self.y = y.flatten()

        # Real offshore measurements in the North Sea, 365 days in total with 10 min interval.
        d = np.load(example_data_path + "/time_series.npz")
        self.day_start = np.random.randint(0, high=300, size=None, dtype='l')
        self.n_days = 1
        wd, ws, ws_std = [d[k][6 * 24 * self.day_start:6 * 24 * (self.n_days + self.day_start)] for k in
                          ['wd', 'ws', 'ws_std']]
        ti = np.minimum(ws_std / ws, .5)

        wd = wd / 360 * 60 + 240
        ws = ws / 25 * 5 + 8

        self.j = 0
        self.Tf = np.size(ws, 0)
        self.N = len(self.x)
        self.action_dim = self.N
        self.observation_dim = self.N * 6
        self.max_action = 40
        self._max_episode_steps = self.Tf

        self.wind_direction_profile = np.reshape(wd, [self.Tf, 1])
        self.wind_speed_profile = np.reshape(ws, [self.Tf, 1])
        self.turbulence_profile = np.reshape(ti, [self.Tf, 1])
        self.time_stamp = np.arange(len(wd)) / 6 / 24

        if evaluation:
            print('Evaluation activated: wind direction and wind speed are fixed')
            # keep wind speed and direction constant during an evaluation
            self.wind_direction_profile = 270 * np.ones([self.Tf, 1])
            self.wind_speed_profile = 10 * np.ones([self.Tf, 1])
            self.turbulence_profile = 0.06 * np.ones([self.Tf, 1])

        # for visualize
        self.horizontal_planes_set = []
        self.y_planes_set = []

        # data collection
        self.power_profile = np.zeros([self.Tf, self.N])
        self.PtfmSurge_profile = np.zeros([self.Tf, self.N])
        self.PtfmSway_profile = np.zeros([self.Tf, self.N])
        self.PtfmPitch_profile = np.zeros([self.Tf, self.N])
        self.PtfmYaw_profile = np.zeros([self.Tf, self.N])
        self.Yaw_angles_profile = np.zeros([self.Tf, self.N])
        self.action_profile = np.zeros([self.Tf, self.N])
        self.previous_position = np.zeros([4, self.N])
        self.position = np.zeros([4, self.N])

        self.moor_look_up_table = moor_matrix()
        self.moor_look_up_table.get_mooring_matrix(visualize=False)

    def reset(self, visualize):

        # simulation setup
        self.fmodel = FlorisModel(r"..\inputs_floating\emgauss_floating.yaml")
        self.j = 0

        # for visualize
        self.horizontal_planes_set = []
        self.y_planes_set = []

        # data collection
        self.power_profile = np.zeros([self.Tf, self.N])
        self.PtfmSurge_profile = np.zeros([self.Tf, self.N])
        self.PtfmSway_profile = np.zeros([self.Tf, self.N])
        self.PtfmPitch_profile = np.zeros([self.Tf, self.N])
        self.PtfmYaw_profile = np.zeros([self.Tf, self.N])
        self.Yaw_angles_profile = np.zeros([self.Tf, self.N])
        self.action_profile = np.zeros([self.Tf, self.N])
        self.previous_position = np.zeros([4, self.N])
        self.position = np.zeros([4, self.N])

        # run the first simulation until the convergence of wind turbine position
        x_reposition = copy.deepcopy(self.x)
        y_reposition = copy.deepcopy(self.y)

        self.fmodel.set(
            layout_x=x_reposition,
            layout_y=y_reposition,
            wind_speeds=[self.wind_speed_profile[self.j, 0]],
            wind_directions=[self.wind_direction_profile[self.j, 0]],
            turbulence_intensities=[self.turbulence_profile[self.j, 0]],
        )
        self.fmodel.set(turbine_type=['iea_15MW'])
        self.N = self.fmodel.n_turbines
        self.fmodel.core.farm.tilt_angles = 10 * np.ones((1, self.fmodel.n_turbines))

        Yaw_angles = np.array([16, -14, 7, -7, 0,
                            16, -14, 7, -7, 0,
                            16, -14, 7, -7, 0,
                            16, -14, 7, -7, 0])
        # Yaw_angles = np.zeros([1, 20])
        Yaw_angles = Yaw_angles.reshape([1, self.N])
        self.fmodel.set(yaw_angles= -Yaw_angles)

        for i in range(self.iteration_num):
            self.fmodel.set(layout_x=x_reposition.flatten(), layout_y=y_reposition.flatten())
            self.fmodel.set(yaw_angles= - Yaw_angles + self.position[3] * 60 / np.pi)
            x_reposition = self.x + self.position[0]
            y_reposition = self.y + self.position[1]
            self.fmodel.run()
            if visualize:
                self.render(i)
                print(f'Initialize iteration: {i}')

            wind_speeds = self.fmodel.turbine_average_velocities
            thrust_coefficients = self.fmodel.get_turbine_thrust_coefficients()
            thrusts = 0.5 * 1.225 * (121 ** 2) * np.pi * thrust_coefficients * wind_speeds ** 2

            theta = (-self.wind_direction_profile[self.j, 0] + 270 - Yaw_angles) * np.pi / 180
            self.position = self.moor_look_up_table.get_position(
                (thrusts * np.cos(theta), thrusts * np.sin(theta)))

        # get observation
        wind_directions = self.wind_direction_profile[self.j, 0] * np.ones([1, self.N]) / 360
        turbulence = self.turbulence_profile[self.j, 0] * np.ones([1, self.N]) / 0.2
        obs = np.concatenate([self.fmodel.turbine_average_velocities.flatten() / 25, Yaw_angles.flatten() / 360,
                              self.position[0].flatten() / 200, self.position[1].flatten() / 200,
                              wind_directions.flatten() / 360,
                              turbulence.flatten() / 0.2])
        return obs

    def step(self, action):
        Yaw_angles = np.reshape(action, [1, self.N]) + self.position[3]*180/np.pi

        # set and run the floris model
        x_reposition = self.x + self.position[0]
        y_reposition = self.y + self.position[1]
        self.fmodel.set(layout_x=x_reposition.flatten(), layout_y=y_reposition.flatten())
        self.fmodel.set(
            wind_speeds=[self.wind_speed_profile[self.j, 0]],
            wind_directions=[self.wind_direction_profile[self.j, 0]],
            turbulence_intensities=[self.turbulence_profile[self.j, 0]],
            yaw_angles=Yaw_angles
        )
        self.fmodel.core.farm.tilt_angles = 6 + np.reshape(self.position[2], [1, self.N])
        self.fmodel.run()

        # calculate the reposition in the next time step
        wind_speeds = self.fmodel.turbine_average_velocities
        thrust_coefficients = self.fmodel.get_turbine_thrust_coefficients()
        thrusts = 0.5 * 1.225 * (121 ** 2) * np.pi * thrust_coefficients * wind_speeds ** 2

        theta = (-self.wind_direction_profile[self.j, 0] + 270 - Yaw_angles) * np.pi / 180
        self.position = self.moor_look_up_table.get_position(
            (thrusts * np.cos(theta), thrusts * np.sin(theta)))

        # get next observations
        self.j = self.j + 1
        wind_directions = self.wind_direction_profile[self.j, 0] * np.ones([1, self.N]) / 360
        turbulence = self.turbulence_profile[self.j, 0] * np.ones([1, self.N]) / 0.2
        next_obs = np.concatenate([self.fmodel.turbine_average_velocities.flatten() / 25, Yaw_angles.flatten() / 360,
                                   self.position[0].flatten() / 200, self.position[1].flatten() / 200,
                                   wind_directions.flatten() / 360,
                                   turbulence.flatten() / 0.2])

        power = self.fmodel.get_turbine_powers().flatten() / 1e6
        wind_speed = self.fmodel.turbine_average_velocities
        reward = np.sum((power / 15) ** 2) + 0.5 * np.sum((wind_speed / 10) ** 2)
        # reward = np.sum((power / 15) ** 2)
        info = self.fmodel.get_turbine_powers().flatten() / 1e6
        done = 0
        if self.j == self.Tf - 1:
            done = 1

        return [next_obs, reward, done, info]

    def render(self, i):
        self.horizontal_planes_set.append(
            self.fmodel.calculate_horizontal_plane(
                x_resolution=200,
                y_resolution=100,
                height=150,
            )
        )
        self.y_planes_set.append(
            self.fmodel.calculate_y_plane(
                x_resolution=200,
                z_resolution=100,
                crossstream_dist=0.0,
            )
        )

        # get wind speed, power, thrust
        u_points = self.fmodel.core.flow_field.u
        print(f'wind distribution points{u_points.shape}')
        print(f'wind speed {self.fmodel.turbine_average_velocities}')
        print(f'Platform reposition: surge  {self.position[0]}')
        print(f'Platform reposition: sway  {self.position[1]}')
        print(f'Platform reposition: pitch  {self.position[2]}')
        print(f'Platform reposition: yaw  {self.position[3]}')

        power = self.fmodel.get_turbine_powers().flatten() / 1e6
        print(f'wind turbine power {power},  \n wind farm power {np.sum(power)}')

        # Create the plots
        # fig, ax_list = plt.subplots(2, 1, figsize=(10, 8))
        # ax_list = ax_list.flatten()
        # flowviz.visualize_cut_plane(self.horizontal_planes_set[i], ax=ax_list[0], title="Horizontal")
        # flowviz.visualize_cut_plane(self.y_planes_set[i], ax=ax_list[1], title="Streamwise profile")
        # fig.suptitle("Floating farm")
        # plt.show()
        turbine_names = ["WT-1", "WT-2", "WT-3", "WT-4", "WT-5",
                         "WT-6", "WT-7", "WT-8", "WT-9", "WT-10",
                         "WT-11", "WT-12", "WT-13", "WT-14", "WT-15",
                         "WT-16", "WT-17", "WT-18", "WT-19", "WT-20"]
        fig, ax_list = plt.subplots()
        flowviz.visualize_cut_plane(self.horizontal_planes_set[i], ax=ax_list,
                                    label_contours=False, cmap='Spectral', )
        # 'Diverging',
        #                      ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
        #                       'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'Spectral'])
        layoutviz.plot_turbine_rotors(self.fmodel, ax=ax_list)
        layoutviz.plot_turbine_labels(self.fmodel, ax=ax_list, turbine_names=turbine_names)
        plt.xlabel('X-coordinate(m)', fontsize=15)
        plt.ylabel('Y-coordinate(m)', fontsize=15)
        plt.tick_params(labelsize=15)

        u_at_points = self.fmodel.sample_flow_at_points(np.array([146]), np.array([-6.8]), np.array([150]))

        if i == self.iteration_num-1:

            plt.savefig('./figures/Spectral_yaw_16deg_2.svg', dpi=600)
            print(u_at_points)
            print('savefig already')
            plt.show()

    def save(self, name):

        dict_data = {'V_profile': self.power_profile, 'action_profile': self.action_profile,
                     'PtfmSurge_profile': self.PtfmSurge_profile, 'PtfmSway_profile': self.PtfmSway_profile,
                     'PtfmPitch_profile': self.PtfmPitch_profile, 'PtfmYaw_profile': self.PtfmYaw_profile}

        np.save(name, dict_data)


if __name__ == "__main__":
    env = Environment(evaluation=True)
    print(f'starting from {env.day_start} days')
    observation = env.reset(visualize=True)
    epi_reward = 0
    for j in range(env.Tf - 1):
        actions = np.random.uniform(-30, 30, [1, env.N])
        actions = np.array([16, -14, 7, -7, 0,
                            16, -14, 7, -7, 0,
                            16, -14, 7, -7, 0,
                            16, -14, 7, -7, 0])
        next_obs, reward, done, info = env.step(actions)
        epi_reward = epi_reward + reward
        print(f'Time Step: {env.j}, wind farm power: {np.sum(info)}, reward: {reward}')
    env.render(3)
    print(f'episode reward: {epi_reward}')
