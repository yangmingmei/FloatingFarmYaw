import numpy as np

from floris.type_dec import (
    floris_float_type,
    NDArrayFloat,
)


def pshape(array: np.ndarray, label: str = ""):
    print(label, np.shape(array))


def cosd(angle):
    """
    Cosine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.cos(np.radians(angle))


def sind(angle):
    """
    Sine of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.sin(np.radians(angle))


def tand(angle):
    """
    Tangent of an angle with the angle given in degrees.

    Args:
        angle (float): Angle in degrees.

    Returns:
        float
    """
    return np.tan(np.radians(angle))


def wind_delta(wind_directions: NDArrayFloat | float):
    """
    This function calculates the deviation from West (270) for a given wind direction or series
    of wind directions. First, 270 is subtracted from the input wind direction, and then the
    remainder after dividing by 360 is retained (modulo). The table below lists examples of
    results.

    | Input | Result |
    | ----- | ------ |
    | 270.0 | 0.0    |
    | 280.0 | 10.0   |
    | 360.0 | 90.0   |
    | 180.0 | 270.0  |
    | -10.0 | 80.0   |
    |-100.0 | 350.0  |

    Args:
        wind_directions (NDArrayFloat | float): A single or series of wind directions. They can be
        any number, negative or positive, but it is typically between 0 and 360.

    Returns:
        NDArrayFloat | float: The delta between the given wind direction and 270 in positive
        quantities between 0 and 360. The returned type is the same as wind_directions.
    """

    return (wind_directions - 270) % 360


def rotate_reposition_coordinates_rel_west(
        wind_directions,
        coordinates,
        x_reposition,
        y_reposition,
        x_center_of_rotation=None,
        y_center_of_rotation=None
):
    """
    This function rotates the given coordinates so that they are aligned with West (270) rather
    than North (0). The rotation happens about the centroid of the coordinates.

    Args:
        wind_directions (NDArrayFloat): Series of wind directions to base the rotation.
        coordinates (NDArrayFloat): Series of coordinates to rotate with shape (N coordinates, 3)
            so that each element of the array coordinates[i] yields a three-component coordinate.
        x_center_of_rotation (float, optional): The x-coordinate for the rotation center of the
            input coordinates. Defaults to None.
        y_center_of_rotation (float, optional): The y-coordinate for the rotational center of the
            input coordinates. Defaults to None.
        x_reposition (NDArrayFloat): Series of coordinates for floating wind turbines
        y_reposition (NDArrayFloat): Series of coordinates for floating wind turbines
    """

    # Calculate the difference in given wind direction from 270 / West
    wind_deviation_from_west = wind_delta(wind_directions)
    wind_deviation_from_west = np.reshape(wind_deviation_from_west, (len(wind_directions), 1))

    # Construct the arrays storing the turbine locations
    x_coordinates, y_coordinates, z_coordinates = coordinates.T

    # Find center of rotation - this is the center of box bounding all of the turbines
    if x_center_of_rotation is None:
        x_center_of_rotation = (np.min(x_coordinates) + np.max(x_coordinates)) / 2
    if y_center_of_rotation is None:
        y_center_of_rotation = (np.min(y_coordinates) + np.max(y_coordinates)) / 2

    # Rotate turbine coordinates about the center
    x_reposition_offset = x_reposition - x_center_of_rotation
    y_reposition_offset = y_reposition - y_center_of_rotation

    x_reposition_rotated = (
            x_reposition_offset * cosd(wind_deviation_from_west)
            - y_reposition_offset * sind(wind_deviation_from_west)
            + x_center_of_rotation
    )

    y_reposition_rotated = (
            x_reposition_offset * sind(wind_deviation_from_west)
            + y_reposition_offset * cosd(wind_deviation_from_west)
            + y_center_of_rotation
    )

    z_coord_rotated = np.ones_like(wind_deviation_from_west) * z_coordinates

    return x_reposition_rotated, y_reposition_rotated, z_coord_rotated, x_center_of_rotation, \
        y_center_of_rotation


def reverse_rotate_coordinates_rel_west(
        wind_directions: NDArrayFloat,
        grid_x: NDArrayFloat,
        grid_y: NDArrayFloat,
        grid_z: NDArrayFloat,
        x_center_of_rotation: float,
        y_center_of_rotation: float
):
    """
    This function reverses the rotation of the given grid so that the coordinates are aligned with
    the original wind direction. The rotation happens about the centroid of the coordinates.

    Args:
        wind_directions (NDArrayFloat): Series of wind directions to base the rotation.
        grid_x (NDArrayFloat): X-coordinates to be rotated.
        grid_y (NDArrayFloat): Y-coordinates to be rotated.
        grid_z (NDArrayFloat): Z-coordinates to be rotated.
        x_center_of_rotation (float): The x-coordinate for the rotation center of the
            input coordinates.
        y_center_of_rotation (float): The y-coordinate for the rotational center of the
            input coordinates.
    """
    # Calculate the difference in given wind direction from 270 / West
    # We are rotating in the other direction
    wind_deviation_from_west = -1.0 * wind_delta(wind_directions)

    # Construct the arrays storing the turbine locations
    grid_x_reversed = np.zeros_like(grid_x)
    grid_y_reversed = np.zeros_like(grid_x)
    grid_z_reversed = np.zeros_like(grid_x)
    for wii, angle_rotation in enumerate(wind_deviation_from_west):
        x_rot = grid_x[wii]
        y_rot = grid_y[wii]
        z_rot = grid_z[wii]

        # Rotate turbine coordinates about the center
        x_rot_offset = x_rot - x_center_of_rotation
        y_rot_offset = y_rot - y_center_of_rotation
        x = (
                x_rot_offset * cosd(angle_rotation)
                - y_rot_offset * sind(angle_rotation)
                + x_center_of_rotation
        )
        y = (
                x_rot_offset * sind(angle_rotation)
                + y_rot_offset * cosd(angle_rotation)
                + y_center_of_rotation
        )
        z = z_rot  # Nothing changed in this rotation

        grid_x_reversed[wii] = x
        grid_y_reversed[wii] = y
        grid_z_reversed[wii] = z

    return grid_x_reversed, grid_y_reversed, grid_z_reversed


def set_grid_reposition(self, x_reposition, y_reposition) -> None:
    """
    Create grid points at each turbine for each wind direction and wind speed in the simulation.
    This creates the underlying data structure for the calculation.

    arrays have shape
    (n wind directions, n wind speeds, n turbines, m grid spanwise, m grid vertically)
    - dimension 1: each wind direction
    - dimension 2: each wind speed
    - dimension 3: each turbine
    - dimension 4: number of points in the spanwise direction (ngrid)
    - dimension 5: number of points in the vertical dimension (ngrid)

    For example
    - x is [
        n wind direction,
        n wind speeds,
        n turbines,
        x-component of the points in the spanwise direction,
        x-component of the points in the vertical direction
    ]
    - y is [
        n wind direction,
        n wind speeds,
        n turbines,
        y-component of the points in the spanwise direction,
        y-component of the points in the vertical direction
    ]

    The x,y,z arrays contain the actual locations in that direction.

    # -   **self.grid_resolution** (*int*, optional): The square root of the number
    #             of points to use on the turbine grid. This number will be
    #             squared so that the points can be evenly distributed.
    #             Defaults to 5.

    If the grid conforms to the sequential solver interface,
    it must be sorted from upstream to downstream

    In a y-z plane on the rotor swept area, the -2 dimension is a column of
    points and the -1 dimension is the row number.
    So the following line prints the 0'th column of the the 0'th turbine's grid:
    print(grid.y_sorted[0,0,0,0,:])
    print(grid.z_sorted[0,0,0,0,:])
    And this line prints a single point
    print(grid.y_sorted[0,0,0,0,0])
    print(grid.z_sorted[0,0,0,0,0])
    Note that the x coordinates are all the same for the rotor plane.

    """
    # TODO: Where should we locate the coordinate system? Currently, its at
    # the foot of the turbine where the tower meets the ground.

    # These are the rotated coordinates of the wind turbines based on the wind direction
    x, y, z, self.x_center_of_rotation, self.y_center_of_rotation = rotate_reposition_coordinates_rel_west(
        self.wind_directions,
        self.turbine_coordinates,
        x_reposition,
        y_reposition,
    )

    # -   **rloc** (*float, optional): A value, from 0 to 1, that determines
    #         the width/height of the grid of points on the rotor as a ratio of
    #         the rotor radius.
    #         Defaults to 0.5.

    # Create the data for the turbine grids
    radius_ratio = 0.5
    disc_area_radius = radius_ratio * self.turbine_diameters / 2
    template_grid = np.ones(
        (
            self.n_findex,
            self.n_turbines,
            self.grid_resolution,
            self.grid_resolution,
        ),
        dtype=floris_float_type
    )
    # Calculate the radial distance from the center of the turbine rotor.
    # If a grid resolution of 1 is selected, create a disc_grid of zeros, as
    # np.linspace would just return the starting value of -1 * disc_area_radius
    # which would place the point below the center of the rotor.
    if self.grid_resolution == 1:
        disc_grid = np.zeros((np.shape(disc_area_radius)[0], 1))
    else:
        disc_grid = np.linspace(
            -1 * disc_area_radius,
            disc_area_radius,
            self.grid_resolution,
            dtype=floris_float_type,
            axis=1
        )
    # Construct the turbine grids
    # Here, they are already rotated to the correct orientation for each wind direction
    _x = x[:, :, None, None] * template_grid

    ones_grid = np.ones(
        (self.n_turbines, self.grid_resolution, self.grid_resolution),
        dtype=floris_float_type
    )
    _y = y[:, :, None, None] + template_grid * (disc_grid[None, :, :, None])
    _z = z[:, :, None, None] + template_grid * (disc_grid[:, None, :] * ones_grid)

    # Sort the turbines at each wind direction

    # Get the sorted indices for the x coordinates. These are the indices
    # to sort the turbines from upstream to downstream for all wind directions.
    # Also, store the indices to sort them back for when the calculation finishes.
    self.sorted_indices = _x.argsort(axis=1)
    self.sorted_coord_indices = x.argsort(axis=1)
    self.unsorted_indices = self.sorted_indices.argsort(axis=1)

    # Put the turbine coordinates into the final arrays in their sorted order
    # These are the coordinates that should be used within the internal calculations
    # such as the wake models and the solvers.
    self.x_sorted = np.take_along_axis(_x, self.sorted_indices, axis=1)
    self.y_sorted = np.take_along_axis(_y, self.sorted_indices, axis=1)
    self.z_sorted = np.take_along_axis(_z, self.sorted_indices, axis=1)

    # Now calculate grid coordinates in original frame (from 270 deg perspective)
    self.x_sorted_inertial_frame, self.y_sorted_inertial_frame, self.z_sorted_inertial_frame = \
        reverse_rotate_coordinates_rel_west(
            wind_directions=self.wind_directions,
            grid_x=self.x_sorted,
            grid_y=self.y_sorted,
            grid_z=self.z_sorted,
            x_center_of_rotation=self.x_center_of_rotation,
            y_center_of_rotation=self.y_center_of_rotation,
        )


if __name__ == "__main__":
    x = [4193.62372567192,
         4586.23169601483, 4207.909022918064, 3822.671339711839,
         3439.185657229091, 3061.6924881018426, 2676.475469958829,
         3868.77479147359, 3479.8198869796765, 3094.605331044565,
         2707.6286696999778, 2315.2792370310417, 1931.7403314094568, 1544.76367006487,
         3525.49397590361, 3133.125437405249, 2740.608840334446, 2350.7386279644306,
         1957.3645790887745, 1564.1385887855513, 1172.5534728058212, 786.11306765524,
         2804.60426320667, 2408.246264201987, 2009.1108340703724, 1611.699404782204,
         1214.816882330398, 819.5326086704703, 422.23540315107
         ]

    y = [1057.3791821561301,
         1672.9962825278799, 2045.790730432492, 2425.399143133105,
         2803.2811561519584, 3175.258221181207, 3554.846270786033,
         1349.4126394052, 1718.7113456882025, 2084.458725409041,
         2451.879160901989, 2824.400849718425, 3188.557259674342, 3555.9776951672898,
         650.60966542751, 1010.7905655699709, 1371.1073784186349, 1728.9949006129034,
         2090.098824895329, 2451.066836471561, 2810.5285815333623, 3165.26765799256,
         327.02602230483, 677.2402547409571, 1029.9085713956556, 1381.0535943774057,
         1731.7312855475302, 2080.9967951159333, 2432.04089219331
         ]
    phi = - 27 * np.pi / 180
    # clockwise rotation -27 degrees and then scale up
    x_reposition = (np.array(x) * np.cos(phi) - np.array(y) * np.sin(phi)) * 240 / 126 - 2200
    y_reposition = (np.array(x) * np.sin(phi) + np.array(y) * np.cos(phi)) * 240 / 126
    z = np.ones_like(x_reposition) * 150

    wind_directions = np.arange(250, 330, 3.0)
    coordinate = np.concatenate([x_reposition.reshape(-1, 1), y_reposition.reshape(-1, 1), z.reshape(-1, 1)], 1)

    x, y, z, x_center_of_rotation, y_center_of_rotation = rotate_coordinates_rel_west(
        wind_directions,
        coordinate,
    )

    x_tile = np.tile(x_reposition.reshape(-1, 1), len(wind_directions))
    x_tile = x_tile.T
    y_tile = np.tile(y_reposition.reshape(-1, 1), len(wind_directions))
    y_tile = y_tile.T

    x_1, y_1, z_1, x_center_of_rotation_1, y_center_of_rotation_1 = rotate_reposition_coordinates_rel_west(
        wind_directions,
        coordinate,
        x_tile,
        y_tile,
    )

    a = 1
