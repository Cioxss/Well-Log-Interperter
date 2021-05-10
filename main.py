from Interperter.loginterperter import Logview
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
test = Logview(r"FEP.xls", 'DE3 Ascii')

test.plotPorosityVsDensity()
test.plotSonicVsDensity()
test.getArchie()
test.calculateVshaleGR()
test.getIndonesia("SP")
test.getPorePermeability()

# print(" ")
# print("Harmonic mean")
# print(statistics.harmonic_mean(test.pore_permeability))
# print("Arhitmetic mean")
# print(np.mean(test.pore_permeability))

#
# print(np.mean(test.pore_permeability))
# quit(1)

l2 = np.where(test.Depth == 1930)[0][0]+1
l3 = np.where(test.Depth == 1935)[0][0]+1
l4 = np.where(test.Depth == 1955)[0][0]+1
l5 = np.where(test.Depth == 1970)[0][0]+1
l6 = np.where(test.Depth == 2000)[0][0]+1
l7 = np.where(test.Depth == 2030)[0][0]+1

a = test.getPorosityCut()
b = test.getPermeabilityCut()
e = test.getShaleCut(test.VshaleGR)
cutoff = e*a*b
print("N/G")
print(np.sum(cutoff[0:l2])/np.size(cutoff[0:l2]))
print(np.sum(cutoff[l2:l3])/np.size(cutoff[l2:l3]))
print(np.sum(cutoff[l3:l4])/np.size(cutoff[l3:l4]))
print(np.sum(cutoff[l4:l5])/np.size(cutoff[l4:l5]))
print(np.sum(cutoff[l5:l6])/np.size(cutoff[l5:l6]))
print(np.sum(cutoff[l6:l7])/np.size(cutoff[l6:l7]))
print(np.sum(cutoff[l7:])/np.size(cutoff[l7:]))
print(np.sum(cutoff)/np.size(cutoff))




print(" ")
print("Effective Porosity")
porosity1 = test.effective_porosity[0:l2, 3]
print(np.mean(porosity1[porosity1 > 0.0001]))
porosity2 = test.effective_porosity[l2:l3, 3]
print(np.mean(porosity2[porosity2 > 0.0001]))
porosity3 = test.effective_porosity[l3:l4, 3]
print(np.mean(porosity3[porosity3 > 0.0001]))
porosity4 = test.effective_porosity[l4:l5, 3]
print(np.mean(porosity4[porosity4 > 0.0001]))
porosity5 = test.effective_porosity[l5:l6, 3]
print(np.mean(porosity5[porosity5 > 0.0001]))
porosity6 = test.effective_porosity[l6:l7, 3]
print(np.mean(porosity6[porosity6 > 0.0001]))
porosity7 = test.effective_porosity[l7:, 3]
print(np.mean(porosity7[porosity7 > 0.0001]))
porosity_avg = test.effective_porosity[:, 3]
print("AVG: ", np.mean(porosity_avg[porosity_avg > 0.0001]))
temporary_array = cutoff*porosity_avg
print("AVG NET: ", np.mean(temporary_array[temporary_array > 0.0001]))

print(" ")
print("Permeability")
print(np.mean(test.pore_permeability[0:l2]))
print(np.mean(test.pore_permeability[l2:l3]))
print(np.mean(test.pore_permeability[l3:l4]))
print(np.mean(test.pore_permeability[l4:l5]))
print(np.mean(test.pore_permeability[l5:l6]))
print(np.mean(test.pore_permeability[l6:l7]))
print(np.mean(test.pore_permeability[l7:]))
print("AVG: ", np.mean(test.pore_permeability))
temporary_array = cutoff*test.pore_permeability
print("AVG NET: ", np.mean(temporary_array[temporary_array > 0.00001]))
print(" ")
print("Vsh SP")
print(np.mean(test.VshaleSP[0:l2]))
print(np.mean(test.VshaleSP[l2:l3]))
print(np.mean(test.VshaleSP[l3:l4]))
print(np.mean(test.VshaleSP[l4:l5]))
print(np.mean(test.VshaleSP[l5:l6]))
print(np.mean(test.VshaleSP[l6:l7]))
print(np.mean(test.VshaleSP[l7:]))
print("AVG: ", np.mean(test.VshaleSP))
temporary_array = cutoff*test.VshaleSP
print("AVG NET: ", np.mean(temporary_array[temporary_array > 0.00001]))

print(" ")
print("Vsh GR")
print(np.mean(test.VshaleGR[0:l2]))
print(np.mean(test.VshaleGR[l2:l3]))
print(np.mean(test.VshaleGR[l3:l4]))
print(np.mean(test.VshaleGR[l4:l5]))
print(np.mean(test.VshaleGR[l5:l6]))
print(np.mean(test.VshaleGR[l6:l7]))
print(np.mean(test.VshaleGR[l7:]))
print("AVG: ", np.mean(test.VshaleGR))
temporary_array = cutoff*test.VshaleGR
print("AVG NET: ", np.mean(temporary_array[temporary_array > 0.00001]))


print(" ")
print("Sw SP Indonesia")
print(np.mean(test.Indonesia_Sw[0:l2]))
print(np.mean(test.Indonesia_Sw[l2:l3]))
print(np.mean(test.Indonesia_Sw[l3:l4]))
print(np.mean(test.Indonesia_Sw[l4:l5]))
print(np.mean(test.Indonesia_Sw[l5:l6]))
print(np.mean(test.Indonesia_Sw[l6:l7]))
print(np.mean(test.Indonesia_Sw[l7:]))
print("AVG: ", np.mean(test.Indonesia_Sw))
temporary_array = cutoff*test.Indonesia_Sw
print("AVG NET: ", np.mean(temporary_array[temporary_array > 0.00001]))

test.getIndonesia("GR")
print(" ")
print("Sw GR Indonesia")
print(np.mean(test.Indonesia_Sw[0:l2]))
print(np.mean(test.Indonesia_Sw[l2:l3]))
print(np.mean(test.Indonesia_Sw[l3:l4]))
print(np.mean(test.Indonesia_Sw[l4:l5]))
print(np.mean(test.Indonesia_Sw[l5:l6]))
print(np.mean(test.Indonesia_Sw[l6:l7]))
print(np.mean(test.Indonesia_Sw[l7:]))
print("AVG: ", np.mean(test.Indonesia_Sw))
temporary_array = cutoff*test.Indonesia_Sw
print("AVG NET: ", np.mean(temporary_array[temporary_array > 0.00001]))


test.plotSonicVsDensity()

print(" ")
print("Effective Porosity")
porosity1 = test.effective_porosity[0:l2, 3]
print(np.mean(porosity1[porosity1 > 0.0001]))
porosity2 = test.effective_porosity[l2:l3, 3]
print(np.mean(porosity2[porosity2 > 0.0001]))
porosity3 = test.effective_porosity[l3:l4, 3]
print(np.mean(porosity3[porosity3 > 0.0001]))
porosity4 = test.effective_porosity[l4:l5, 3]
print(np.mean(porosity4[porosity4 > 0.0001]))
porosity5 = test.effective_porosity[l5:l6, 3]
print(np.mean(porosity5[porosity5 > 0.0001]))
porosity6 = test.effective_porosity[l6:l7, 3]
print(np.mean(porosity6[porosity6 > 0.0001]))
porosity7 = test.effective_porosity[l7:, 3]
print(np.mean(porosity7[porosity7 > 0.0001]))
porosity_avg = test.effective_porosity[:, 3]
print("AVG: ", np.mean(porosity_avg[porosity_avg > 0.0001]))
temporary_array = cutoff*porosity_avg
print("AVG NET: ", np.mean(temporary_array[temporary_array > 0.0001]))

# a = test.getPorosityCut()
# b = test.getPermeabilityCut()
# e = test.getShaleCut(test.VshaleGR)
#
# d = test.Depth*a*b*e


# test.pore_permeability[test.pore_permeability >= 0.5] = 1.5
# test.pore_permeability[test.pore_permeability <= 0.5] = 1
# test.plotSingle(test.pore_permeability, "Permeability", "mD")

#
# test.plotSingle(test.VshaleGR, "Shale Volume GR")
# test.plotSingle(test.VshaleSP, "Shale Volume SP")
