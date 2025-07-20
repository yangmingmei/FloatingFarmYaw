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
        self.fmodel = FlorisModel(r"Inputs/emgauss_floating.yaml")
        self.fmodel.set(turbine_type=['iea_15MW'])
        FlorisModel.assign_hub_height_to_ref_height(self.fmodel)

        self.iteration_num = 5
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
        self.evaluation = evaluation

        if self.evaluation:
            print('Evaluation activated: wind direction and wind speed are fixed')

        # for visualize
        self.horizontal_planes_set = []
        self.y_planes_set = []

        # data collection for plot
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

        #
        d = np.load(example_data_path + "/time_series.npz")
        self.day_start = np.random.randint(0, high=300, size=None, dtype='l')
        self.n_days = 1
        wd, ws, ws_std = [d[k][6 * 24 * self.day_start:6 * 24 * (self.n_days + self.day_start)] for k in
                          ['wd', 'ws', 'ws_std']]

        # Turbulence intensity from field measurements
        ti = np.minimum(ws_std / ws, .5)

        # Ensure the system is a Markov Decision Process (this is essential)
        wd = wd / 360 * 0 + np.random.uniform(-20, 20) * 0 + 270
        ws = ws / 25 * 0 + np.random.uniform(8, 11)

        self.wind_direction_profile = np.reshape(wd, [self.Tf, 1])
        self.wind_speed_profile = np.reshape(ws, [self.Tf, 1])
        self.turbulence_profile = np.reshape(ti, [self.Tf, 1])
        self.time_stamp = np.arange(len(wd)) / 6 / 24

        if self.evaluation:
            # keep wind speed and direction constant during an evaluation
            self.wind_direction_profile = 270 * np.ones([self.Tf, 1])
            self.wind_speed_profile = 10 * np.ones([self.Tf, 1])
            self.turbulence_profile = 0.10 * np.ones([self.Tf, 1])
            self.day_start = None

        # Simulation setup
        self.j = 0

        # For visualize
        self.horizontal_planes_set = []
        self.y_planes_set = []

        # Data collection for plot
        self.power_profile = np.zeros([self.Tf, self.N])
        self.PtfmSurge_profile = np.zeros([self.Tf, self.N])
        self.PtfmSway_profile = np.zeros([self.Tf, self.N])
        self.PtfmPitch_profile = np.zeros([self.Tf, self.N])
        self.PtfmYaw_profile = np.zeros([self.Tf, self.N])
        self.Yaw_angles_profile = np.zeros([self.Tf, self.N])
        self.action_profile = np.zeros([self.Tf, self.N])
        self.previous_position = np.zeros([4, self.N])
        self.position = np.zeros([4, self.N])

        # Run the first simulation until the convergence of wind turbine position
        x_reposition = copy.deepcopy(self.x)
        y_reposition = copy.deepcopy(self.y)

        self.fmodel.set(
            layout_x=x_reposition,
            layout_y=y_reposition,
            wind_speeds=[self.wind_speed_profile[self.j, 0]],
            wind_directions=[self.wind_direction_profile[self.j, 0]],
            turbulence_intensities=[self.turbulence_profile[self.j, 0]],
        )

        self.N = self.fmodel.n_turbines
        self.fmodel.core.farm.tilt_angles = 6 * np.ones((1, self.fmodel.n_turbines))

        Yaw_angles = np.array([16, -14, 7, -7, 0,
                               16, -14, 7, -7, 0,
                               16, -14, 7, -7, 0,
                               16, -14, 7, -7, 0])
        # Yaw_angles = np.array([-21.309078, 23.364592, 3.8631878, - 26.064625, 16.051409, 21.02423,
        #                        - 24.655594, - 4.2792993, 25.919191, - 14.875843, - 22.100471, 18.923973,
        #                        - 19.735582, 22.962261, - 7.8848705, - 22.344395, 18.467663, - 19.431631,
        #                        22.612972, - 11.114875])
        # Yaw_angles = np.array([-22.590303, 19.78203, - 17.770546, 23.451992, - 10.3017, 21.510345,
        #                        - 19.543684, 18.991922, - 21.928532, 12.345204, - 22.587929, 18.460304,
        #                        - 19.628523, 21.331757, - 11.954666, 22.294044, - 20.606354, 15.422161,
        #                        - 25.18711, 10.073893])
        # Yaw_angles = np.zeros([1, self.N])
        Yaw_angles = Yaw_angles.reshape([1, self.N])
        self.fmodel.set(yaw_angles=-Yaw_angles)

        for i in range(self.iteration_num):
            self.fmodel.set(layout_x=x_reposition.flatten(), layout_y=y_reposition.flatten())
            self.fmodel.set(yaw_angles=- Yaw_angles + self.position[3] * 180 / np.pi)
            x_reposition = self.x + self.position[0]
            y_reposition = self.y + self.position[1]
            self.fmodel.run()
            if visualize:
                print(f'Initialize iteration: {i}')
                self.render(i)

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
        Yaw_angles = np.reshape(action, [1, self.N]) + self.position[3] * 180 / np.pi

        # set and run the floris model
        for i in range(self.iteration_num):
            x_reposition = self.x + self.position[0]
            y_reposition = self.y + self.position[1]
            self.fmodel.set(layout_x=x_reposition.flatten(), layout_y=y_reposition.flatten())
            self.fmodel.set(yaw_angles=- Yaw_angles + self.position[3] * 60 / np.pi)
            self.fmodel.run()

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
        wind_speeds = self.fmodel.turbine_average_velocities

        V_0 = self.wind_speed_profile[self.j, 0]
        P_0 = 0.5 * 0.48 * 1.225 * np.pi * (120 ** 2) * (V_0 ** 3) / 1e6
        if P_0 > 15:
            P_0 = 15
        k_1 = (np.cos((self.wind_direction_profile[self.j, 0] - 270) * np.pi / 180) ** 3) * (10 / min(V_0, 12)) ** 1.5
        k_2 = 0
        reward = (np.sum((power / P_0) ** 2) * k_1) + (np.sum((wind_speeds / V_0) ** 2) * k_2) - 8

        info = self.fmodel.get_turbine_powers().flatten() / 1e6
        done = 0
        if self.j == self.Tf - 1:
            done = 1

        return [next_obs, reward, done, info]

    def render(self, i):
        # i :iteration steps
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
        print(f'wind speed {self.fmodel.turbine_average_velocities}')
        print(f'Platform reposition: surge  {self.position[0]}')
        print(f'Platform reposition: sway  {self.position[1]}')
        print(f'Platform reposition: pitch  {self.position[2]}')
        print(f'Platform reposition: yaw  {self.position[3]}')
        power = self.fmodel.get_turbine_powers().flatten() / 1e6
        print(f'wind turbine power {power},  \n wind farm power {np.sum(power)}')
        print(f'\n')

        turbine_names = ["WT-1", "WT-2", "WT-3", "WT-4", "WT-5",
                         "WT-6", "WT-7", "WT-8", "WT-9", "WT-10",
                         "WT-11", "WT-12", "WT-13", "WT-14", "WT-15",
                         "WT-16", "WT-17", "WT-18", "WT-19", "WT-20"]
        fig, ax_list = plt.subplots()
        flowviz.visualize_cut_plane(self.horizontal_planes_set[i], ax=ax_list,
                                    label_contours=False, cmap='Spectral')

        layoutviz.plot_turbine_rotors(self.fmodel, ax=ax_list)
        layoutviz.plot_turbine_labels(self.fmodel, ax=ax_list, turbine_names=turbine_names)
        plt.xlabel('X-coordinate(m)', fontsize=15)
        plt.ylabel('Y-coordinate(m)', fontsize=15)
        plt.tick_params(labelsize=15)

        if i == self.iteration_num - 1:
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
        _, Reward, _, Power = env.step(actions)
        epi_reward = epi_reward + Reward
        print(f'Time Step: {env.j}, wind farm power: {np.sum(Power)}, reward: {Reward}')
    print(f'episode reward: {epi_reward}')
