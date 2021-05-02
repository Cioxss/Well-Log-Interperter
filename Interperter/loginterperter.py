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

    plt.plot([-1.7*0.01, 0.25], [2.657, 2.4], "k-")         # shale
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
    shale_points[:, 3] = shale_points[:, 3]*0 # Points below the shale line
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

    def __init__(self, filedirectory, sheet, uselessrows = 0,):
        self.sheet = sheet
        self.Datafile = pd.read_excel(filedirectory, sheet_name=sheet, skiprows = 47)
        #print(Datafile)
        self.GR = (self.Datafile["GR"])[uselessrows:].copy()
        self.Depth = (self.Datafile["Depth"])[uselessrows:].copy()
        self.DENS = (self.Datafile["FDC"])[uselessrows:].copy()
        self.NEUT = (self.Datafile["NESNP"])[uselessrows:].copy()
        self.RESD = (self.Datafile["RST"])[uselessrows:].copy()
        self.SP = (self.Datafile["SP"])[uselessrows:].copy()
        self.SON = (self.Datafile["SON"])[uselessrows:].copy()
        if sheet == "DE1 Ascii":
            self.PERM = (self.Datafile["PERMCOR"])[uselessrows:].copy()
        self.NEUT = self.NEUT * 0.01
        self.trimData()



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

    def getKroom(self):
        self.insitu_permeability = 0.5689 * self.PERM - 0.3279

    def calculateVshaleGR(self):
        self.GRmin = 19
        self.GRmax = 96
        self.VshaleGR = (self.GR - self.GRmin) / (self.GRmax - self.GRmin)

    def calculateVshaleSP(self):
        self.SPcl = -21
        self.SPsh = -6
        self.VshaleSP = (self.SP - self.SPcl) / (self.SPsh - self.SPcl)

    def getArchie(self):
        self.R_w = 0.028
        self.m = 1.85
        self.n = 2
        self.Archie_Sw = (self.R_w / (((self.effective_porosity[:, 3]*0.01) ** self.m) * self.RESD))**(1/self.n)

    def getIndonesia(self, method):
        self.R_w = 0.028
        self.m = 1.85
        self.n = 2
        self.Rsh = 1
        numerator = np.sqrt(1/self.RESD)
        Vshale = 0
        if method == "GR":
            self.calculateVshaleGR()
            Vshale = self.VshaleGR
        elif method == "SP":
            self.calculateVshaleSP()
            Vshale = self.VshaleSP
        else:
            print("Only GR and SP shale volumes are supported at this moment.")
            quit(1)
        denomenator_1 = (Vshale ** (1 - Vshale) / np.sqrt(self.Rsh))
        denomenator_2 = np.sqrt((self.effective_porosity[:, 3]*0.01) ** self.m / self.R_w)
        self.Indonesia_Sw = (numerator / (denomenator_1 + denomenator_2)) ** (2 / self.n)

    def getTimur(self, method):

        if method == "Archie":
            S_w = self.Archie_Sw
        elif method == "Indonesia":
            S_w = self.Indonesia_Sw
        else:
            print("Only Archie and Indonesia are supported at this moment.")
            quit(1)

        self.permeability = (1e4*(self.effective_porosity[:, 3]/100)**4.5)/(S_w**2)

    def calcMeanVshale(self):
        self.calculateVshaleGR()
        self.calculateVshaleSP()
        self.meanVshale = (self.VshaleGR + self.VshaleSP) * 1/2

    def plotSingle(self, plot_variable_name, name="None", xname="None"):

        plt.rcParams['figure.figsize'] = [5, 10]
        plt.plot(plot_variable_name, self.Depth)
        plot_variable_name[plot_variable_name > 1e200] = 0
        plt.title(name)
        plt.ylabel("Depth (m)")
        plt.xlabel(xname)
        #plt.xlim(min(plot_variable_name), max(plot_variable_name))
        #plt.xlim(-1, 2)
        plt.ylim(max(self.Depth), min(self.Depth))
        plt.show()

    def plotDouble(self, plot_variable_name, plot_variable_name2, name="None"):

        plt.rcParams['figure.figsize'] = [5, 10]
        plt.plot(plot_variable_name, self.Depth)
        plt.plot(plot_variable_name2, self.Depth)
        plt.title(name)
        plt.ylabel("Depth (m)")
        plt.xlabel(name)
        plt.xlim(np.nanmin([plot_variable_name]), np.nanmax(plot_variable_name))
        plt.ylim(max(self.Depth), min(self.Depth))
        plt.show()

    def getPorePermeability(self):
        self.pore_permeability = 0.2923*np.exp(0.2176*self.effective_porosity[:, 3])

    def getPorosityCut(self):
        cut_porosity = np.where(self.effective_porosity[:, 3]/100 >= 0.02, 1, 0)
        return cut_porosity

    def getSaturationCut(self, saturation):
        cut_saturation = np.where(saturation <= 0.6, 1, 0)
        return cut_saturation

    def getShaleCut(self, volumes):
        cut_shale = np.where(volumes <= 0.4, 1, 0)
        return cut_shale

    def getPermeabilityCut(self):
        cut_permeability = np.where(self.pore_permeability >= 0.5, 1, 0)
        return cut_permeability

    def plotPorosityVsDensity(self):

        #plotting pure water filled sandstone and gas filled points
        sorted_eff = getSS(self.trimmed_DEPTH, self.trimmed_DENS, self.trimmed_NEUT)
        sorted_eff[:, 2] = sorted_eff[:, 2] * 100
        sorted_eff[:, 3] = sorted_eff[:, 3] * 100
        meow_excel = pd.DataFrame(sorted_eff)
        meow_excel.columns = ["Depth", "Density", "Neutron", "Effective"]
        sheet_name = self.sheet + "_effectiveporosity.xls"
        meow_excel.to_excel(sheet_name, index=False)

        self.effective_porosity = sorted_eff

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
        f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1, 6, sharey=True)
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

        ax5.plot(self.SP, self.Depth, linewidth=lw, color="red")
        ax5.set_title('Spontaneous Potential')
        ax5.set_xlim(np.nanmin(self.SP), np.nanmax(self.SP))
        ax5.set_xlabel('[mV]')

        ax6.plot(self.SON, self.Depth, linewidth=lw, color="red")
        ax6.set_title('Sonic Data')
        ax6.set_xlim(np.nanmin(self.SON), np.nanmax(self.SON))
        ax6.set_xlabel('[micros/feet]')

        plt.show()

