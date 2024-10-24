# Coupling Moorpy to Pywake to simulate floating wind turbines:
# Example of getting a mooring matrix for each floating wind turbine in different water depth.
# surge sway pitch yaw

import numpy as np
import matplotlib.pyplot as plt
import moorpy as mp
import numpy as np
from scipy.interpolate import interpn, griddata
import pandas as pd
from pandas import Series, DataFrame


# OpenFast default
# self.depth = 200  # water depth [m]
# self.angles = np.radians([60, 180, 300])  # line headings list [rad]
# self.rAnchor = 837  # anchor radius/spacing [m] %
# self.lineLength = 850  # line unstretched length [m]


class moor_matrix():
    def __init__(self):
        # ----- choose some system geometry parameters -----
        self.depth = 400  # water depth [m]
        self.angles = np.radians([45, 180, 315])  # line headings list [rad]
        self.rAnchor = 969  # anchor radius/spacing [m] %
        self.lineLength = 1180  # line unstretched length [m]
        self.zFair = -21  # fairlead z elevation [m]
        self.rFair = 20  # fairlead radius [m]

        self.tab_um = 21
        self.FX = np.linspace(-3.5e6, 3.5e6, self.tab_um)
        self.FY = np.linspace(-2.5e6, 2.5e6, self.tab_um)
        self.matrix_xtab = np.zeros([self.tab_um, self.tab_um])
        self.matrix_ytab = np.zeros([self.tab_um, self.tab_um])
        self.matrix_rytab = np.zeros([self.tab_um, self.tab_um])
        self.matrix_rztab = np.zeros([self.tab_um, self.tab_um])

    def get_mooring_matrix(self, visualize):
        """
        Input: mooring parameters
        Output: matrix
        """
        typeName = "chain1"  # identifier string for the line type

        # ----- set up the mooring system and floating body -----

        # Create new MoorPy System and set its depth
        ms = mp.System(depth=self.depth)

        # add a line type
        ms.setLineType(dnommm=120, material='chain', name=typeName)  # this would be 120 mm chain

        # Add a free, body at [0,0,0] to the system (including some properties to make it hydrostatically stiff)
        ms.addBody(0, np.zeros(6), m=20095e3, v=1952, rM=150, AWP=1215)

        # For each line heading, set the anchor point, the fairlead point, and the line itself
        for i, angle in enumerate(self.angles):
            # create end Points for the line
            ms.addPoint(1,
                        [self.rAnchor * np.cos(angle), self.rAnchor * np.sin(angle),
                         -self.depth])  # create anchor point (type 0, fixed)
            ms.addPoint(1,
                        [self.rFair * np.cos(angle), self.rFair * np.sin(angle),
                         self.zFair])  # create fairlead point (type 0, fixed)

            # attach the fairlead Point to the Body (so it's fixed to the Body rather than the ground)
            ms.bodyList[0].attachPoint(2 * i + 2, [self.rFair * np.cos(angle), self.rFair * np.sin(angle), self.zFair])

            # add a Line going between the anchor and fairlead Points
            ms.addLine(self.lineLength, typeName, pointA=2 * i + 1, pointB=2 * i + 2)

        ms.initialize()  # make sure everything's connected

        if visualize:
            ms.solveEquilibrium()  # equilibrate
            fig, ax = ms.plot()  # plot the system in original configuration

        # ----- run the model to demonstrate -----

        for i in range(np.size(self.FX, 0)):
            for j in range(np.size(self.FY, 0)):
                ms.bodyList[0].f6Ext = np.array(
                    [self.FX[i], self.FY[j], 0, 0, self.FX[i] * 145, 0])  # apply an external force on the body
                ms.solveEquilibrium3()  # equilibrate
                # sequence: surge sway heave roll pitch yaw
                self.matrix_xtab[i, j] = ms.bodyList[0].r6[0]
                self.matrix_ytab[i, j] = ms.bodyList[0].r6[1]
                self.matrix_rytab[i, j] = ms.bodyList[0].r6[4]
                self.matrix_rztab[i, j] = ms.bodyList[0].r6[5]
                # print(f"Body offset position is {ms.bodyList[0].r6}")

        if visualize:
            print(f"Body maximum offset position is {ms.bodyList[0].r6}")
            fig, ax = ms.plot(ax=ax, color='red')  # plot the system in original configuration

            plt.show()

        return 0

    def get_position(self, xi):
        points = [self.FX, self.FY]
        surge = interpn(points, self.matrix_xtab, xi)
        sway = interpn(points, self.matrix_ytab, xi)
        pitch = interpn(points, self.matrix_rytab, xi)
        yaw = interpn(points, self.matrix_rztab, xi)

        return [surge, sway, pitch, yaw]


if __name__ == "__main__":
    moor_matrix = moor_matrix()
    moor_matrix.get_mooring_matrix(visualize=True)
    position = moor_matrix.get_position(([3e6, 2e6, 1e6], [2e6, 1e6, 0.75e6]))
    print(f" Body offset position under force [3e6, 2e6, 1e6] and [2e6, 1e6, 0.75e6] is respectively  ")
    print(f"Surge: {position[0]} Sway: {position[1]} pitch: {position[2]} yaw: {position[3]} ")
