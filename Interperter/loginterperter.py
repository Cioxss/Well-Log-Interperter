from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit
def isPointOnLine(line, point):
    line_vec = [line[1][0] - line[0][0], line[1][1] - line[0][1]]
    point_vec = [point[2] - line[0][0], point[1] - line[0][1]]

    cross = line_vec[1] * point_vec[0] - line_vec[0] * point_vec[1]
    return cross
@jit
def checkLine(DENS, line, point_matrix):
    for i in DENS:
        point_matrix[i, 3] = isPointOnLine(line, point_matrix[i, :])
    return point_matrix


def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x, y
    else:
        return False

def findPor(SST, gas_trend, matrix):
    #     line_gas = np.array([[-0.01, 2.08],
    #                          [0.10, 2.2]])
    vector = np.array([[0., 0.], [0., 0.]])
    meow = np.zeros_like(matrix)
    i = 0
    for row in matrix:
        difference = gas_trend[1, :] - gas_trend[0, :]
        vector[0, :] = [row[2], row[1]]
        vector[1, :] = [row[2], row[1]] + difference
        L1 = line(vector[0], vector[1])
        L2 = line(SST[0], SST[1])
        x, y = intersection(L1, L2)
        meow[i, 0] = x
        meow[i, 1] = y
        i += 1
    matrix[:, 3] = calculateEfPor(meow)
    return meow, matrix

def calculateEfPor(matrix):
    densities = matrix[:, 1]
    matrix[:, 2] = densities*-0.664697193501+1.76610044313
    return matrix[:, 2]



def getSS(Depth, DENS, NEUT):
    point_matrix = np.zeros([np.shape(DENS)[0], 4])
    point_matrix[:, 0] = Depth
    point_matrix[:, 1] = DENS
    point_matrix[:, 2] = NEUT
    plt.figure(figsize=(10, 10))
    plt.title("Scatter plot of Density vs Porosity")
    plt.ylabel("Density")
    plt.xlabel("Neutron Porosity")
    plt.gca().invert_yaxis()

    plt.plot([-1.7*0.01, 0.25], [2.657, 2.4], "g-")         # shale
    plt.plot([-1.7*0.01, 40.7*0.01], [2.657, 1.98], "y--")  # SST
    #plt.plot([-0.01, 0.10], [2.08, 2.2])                   # gas trend

    line_sst = np.array([[-1.7*0.01, 2.657], [40.7*0.01, 1.98]])
    line_shale = np.array([[-1.7*0.01, 2.657], [0.25, 2.4]])
    line_gas = np.array([[-0.01, 2.08], [0.10, 2.2]])


    loop_data = np.arange(0, len(DENS))
    point_matrix2 = checkLine(loop_data, line_shale, point_matrix)
    sorted_matrix2 = point_matrix2[point_matrix2[:, 3].argsort()]
    first_zero = (np.where(sorted_matrix2[:, 3] > 0))[0][0]
    shale_points = sorted_matrix2[:first_zero, :]
    shale_points[:, 3] = shale_points[:, 3]*0  # Points below the shale line
    sorted_matrix2 = sorted_matrix2[first_zero:, :]
    loop_data = np.arange(0, len(sorted_matrix2[:, 3]))

    point_matrix3 = checkLine(loop_data, line_sst, sorted_matrix2)
    sorted_matrix3 = point_matrix3[point_matrix3[:, 3].argsort()]
    first_zero = (np.where(sorted_matrix3[:, 3] > 0))[0][0]

    meow, matrix_effpart1 = findPor(line_sst, line_gas, sorted_matrix3[first_zero:, :])
    meow2, matrix_effpart2 = findPor(line_sst, line_shale, sorted_matrix3[:first_zero, :])

    matrix_eff = np.append(matrix_effpart1, matrix_effpart2, axis=0)
    matrix_eff = np.append(matrix_eff, shale_points, axis=0)
    sorted_eff = matrix_eff[matrix_eff[:, 0].argsort()]

    plt.scatter(sorted_matrix3[first_zero:, 2], sorted_matrix3[first_zero:, 1], color="green")
    plt.scatter(sorted_matrix3[:first_zero, 2], sorted_matrix3[:first_zero, 1], color="red")
    plt.scatter(meow[:, 0], meow[:, 1], color="blue")
    plt.scatter(meow2[:, 0], meow2[:, 1], color="yellow")

    plt.show()
    return sorted_eff

class Logview:

    def __init__(self, filedirectory, uselessrows = 0, lb = 0, rb = 0):
        self.Datafile = pd.read_excel(filedirectory, sheet_name='DE1 Ascii', skiprows = 48)
        #print(Datafile)
        self.GR = (self.Datafile["GR"])[uselessrows:].copy()
        self.Depth = (self.Datafile["Depth"])[uselessrows:].copy()
        self.DENS = (self.Datafile["FDC"])[uselessrows:].copy()
        self.NEUT = (self.Datafile["NESNP"])[uselessrows:].copy()
        self.RESD = (self.Datafile["RST"])[uselessrows:].copy()
        self.NEUT = self.NEUT * 0.01
        self.trimData()
        lbid = 0
        if lb != 0 and rb != 0:
            lbid = np.where(np.abs(self.Depth - lb) < 1)
            rbid = np.where(np.abs(self.Depth - rb) < 1)


    def trimData(self):
        valid_range = np.array([[pd.Series.first_valid_index(self.GR), pd.Series.last_valid_index(self.GR)],
                                [pd.Series.first_valid_index(self.Depth), pd.Series.last_valid_index(self.Depth)],
                                [pd.Series.first_valid_index(self.DENS), pd.Series.last_valid_index(self.DENS)],
                                [pd.Series.first_valid_index(self.NEUT), pd.Series.last_valid_index(self.NEUT)],
                                [pd.Series.first_valid_index(self.RESD), pd.Series.last_valid_index(self.RESD)]
                                ])
        min_boundary = np.max(valid_range[:,0])
        max_boundary = np.min(valid_range[:,1])
        self.trimmed_GR = self.GR[min_boundary:max_boundary+1]
        self.trimmed_DEPTH = self.Depth[min_boundary:max_boundary+1]
        self.trimmed_DENS =  self.DENS[min_boundary:max_boundary+1]
        self.trimmed_NEUT = self.NEUT[min_boundary:max_boundary+1]
        self.trimmed_RESD = self.RESD[min_boundary:max_boundary+1]

    def calculateVshale(self):
        self.GRmin = 20
        self.GRmax = 120
        self.Vshale = (self.GR - self.GRmin) / (self.GRmax - self.GRmin)


    def getShaleLine(self, cutoff=0.97):
        ShaleLine = []

        for i in range(500, len(self.Vshale)):
            if(self.Vshale[i] >= cutoff):
                if not (np.isnan(self.DENS[i]) and np.isnan(self.NEUT[i])):
                    ShaleLine.append([self.DENS[i], self.NEUT[i]])
        self.ShaleLine = np.array(ShaleLine)

    def getSandstoneLine(self, cutoff=0.10):
        SandstoneLine = []

        for i in range(len(self.Vshale)):
            if (self.Vshale[i] <= cutoff):
                if not (np.isnan(self.DENS[i]) and np.isnan(self.NEUT[i])):
                    SandstoneLine.append([self.DENS[i], self.NEUT[i]])
        self.SandstoneLine = np.array(SandstoneLine)

    def plotSingle(self, plot_variable_name, name="None"):

        plt.rcParams['figure.figsize'] = [5, 10]
        plt.plot(plot_variable_name, self.Depth)
        plt.title(name)
        plt.ylabel("Depth (m)")
        plt.xlabel(name)
        plt.xlim(np.nanmin(plot_variable_name), np.nanmax(plot_variable_name))
        #plt.ylim(max(self.Depth), min(self.Depth))
        plt.ylim(1500, 600)
        #plt.xlim(min(plot_variable_name), max(plot_variable_name))
        plt.xlim(0, 1)
        plt.show()

    def plotPorosityVsDensity(self):

        #plotting all points
        labels1 = ['SST line',
                  'All measurements',
                  'Low Vshale values (<0.1)',
                  'High Vshale values (>0.95)']

        labels2 = ['SST line',
                  'All measurements',
                  'Pure SST',
                  'Gas SST']

        porosity = np.linspace(0, 1, 1000) - 0.05
        sandstone = 2.65 * (1 - porosity - 0.05) + 1 * porosity + 0.05 + 0.01
        porosity2 = np.linspace(0, 0.4, 100)
        limestone = np.linspace(2.712, 2.167, 100)
        #Plotting curved and corrected sansdtone functions

        dolomiteLine = np.array([
            [0.39086, 2.223],
            [0.35, 2.323],
            [0.30, 2.434],
            [0.25, 2.545],
            [0.20, 2.645],
            [0.15, 2.734],
            [0.10, 2.8],
            [0.05, 2.856],
            [0.02724, 2.867]
        ])

        # porosity3 =
        # shaleBaseLine =

        #plt.plot(sandstoneLine[:,0], sandstoneLine[:, 1], "red")
        #
        # plt.plot(porosity, sandstone, "y--", label=labels2[0])
        # plt.plot(porosity2, limestone, 'm--')
        # plt.plot(dolomiteLine[:, 0], dolomiteLine[:, 1], 'chartreuse')
        # plt.scatter(self.NEUT, self.DENS, label=labels2[1])
        # plt.title("Scatter plot of Density vs Porosity")
        # plt.ylabel("Density")
        # plt.xlabel("Neutron Porosity")
        # plt.gca().invert_yaxis()

        #plotting pure water filled sandstone and gas filled points
        meow_matrix = getSS(self.trimmed_DEPTH, self.trimmed_DENS, self.trimmed_NEUT)
        meow_matrix[:, 2] = meow_matrix[:, 2] * 100
        meow_matrix[:, 3] = meow_matrix[:, 3] * 100
        meow_excel = pd.DataFrame(meow_matrix)
        meow_excel.columns = ["Depth", "Density", "Neutron", "Effective"]
        meow_excel.to_excel("test.xls", index=False)

        quit(1)

        #plotting sadstone / shale points

        plt.scatter(self.SandstoneLine[:, 1], self.SandstoneLine[:, 0], label=labels1[2])
        plt.scatter(self.ShaleLine[:, 1], self.ShaleLine[:, 0], label=labels1[3])
        # plt.scatter(pureSS[:, 1], pureSS[:, 0], label=labels2[2])
        # plt.scatter(gasSS[:, 1], gasSS[:, 0], label=labels2[3])

        #sandstone = 2.65 * (1 - porosity) + 1 * porosity

        plt.legend(labels=labels2)
        plt.xlim(-0.05, 0.4)
        plt.ylim(2.9, 1.95)
        plt.grid()
        plt.show()

    def Plot(self):
        # define minimum and maximum depth for plotting
        # minimum_depth = min(self.Depth)
        minimum_depth = min(self.Depth)
        maximum_depth = max(self.Depth)
        np.nan_to_num(self.DENS)
        # define the linewidth
        lw = 0.4

        # define the figure size
        plt.rcParams['figure.figsize'] = [22, 16]
        # make a figure consisting of 6 subplots that share the depth axis (y)
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
        ax1.set_ylim(maximum_depth, minimum_depth)

        # plot the 6 individual borehole logs
        ax1.plot(self.GR, self.Depth, linewidth=lw, color="green")
        ax1.set_title('Gamma ray')
        ax1.set_xlabel('[API]')
        ax4.set_xlim(min(self.GR), max(self.GR))
        ax1.set_ylabel('depth [m]')

        ax2.plot(self.DENS, self.Depth, linewidth=lw, color="blue")
        ax2.set_title('Density')
        ax2.set_xlim(np.nanmin(self.DENS), np.nanmax(self.DENS))
        ax2.set_xlabel('[g/cm$^3$]')

        ax3.plot(self.NEUT, self.Depth, linewidth=lw, color="black")
        ax3.set_title('Neutron Porosty')
        ax3.set_xlim(np.nanmin(self.NEUT), np.nanmax(self.NEUT))
        ax3.set_xlabel('[-]')

        ax4.plot(self.RESD, self.Depth, linewidth=lw, color="red")
        ax4.set_title('Resistivity')
        ax4.set_xlim(np.nanmin(self.RESD), np.nanmax(self.RESD))
        ax4.set_xlabel('[$\Omega$m]')

        plt.show()

