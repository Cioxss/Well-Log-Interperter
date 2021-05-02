from Interperter.loginterperter import Logview
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

test = Logview(r"FEP.xls", 'DE4 Ascii')

test.plotPorosityVsDensity()

test.getArchie()
test.calculateVshaleGR()
test.getIndonesia("SP")
test.getPorePermeability()

# test.plotSingle(test.effective_porosity[:, 3]/100, "Effective Porosity", "(-)")


l2 = np.where(test.Depth == 1925)[0][0]+1
l3 = np.where(test.Depth == 1935)[0][0]+1
l4 = np.where(test.Depth == 1965)[0][0]+1
l5 = np.where(test.Depth == 2010)[0][0]+1
l6 = np.where(test.Depth == 2025)[0][0]+1
l7 = np.where(test.Depth == 2040)[0][0]+1

a = test.getPorosityCut()
b = test.getPermeabilityCut()
e = test.getShaleCut(test.VshaleGR)
cutoff = a*b*e
print("N/G")
print(np.sum(cutoff[0:l2])/np.size(cutoff[0:l2]))
print(np.sum(cutoff[l2:l3])/np.size(cutoff[l2:l3]))
print(np.sum(cutoff[l3:l4])/np.size(cutoff[l3:l4]))
print(np.sum(cutoff[l4:l5])/np.size(cutoff[l4:l5]))
print(np.sum(cutoff[l5:l6])/np.size(cutoff[l5:l6]))
print(np.sum(cutoff[l6:l7])/np.size(cutoff[l6:l7]))
print(np.sum(cutoff[l7:])/np.size(cutoff[l7:]))
print("Permeability")
print(np.mean(test.pore_permeability[0:l2]))
print(np.mean(test.pore_permeability[l2:l3]))
print(np.mean(test.pore_permeability[l3:l4]))
print(np.mean(test.pore_permeability[l4:l5]))
print(np.mean(test.pore_permeability[l5:l6]))
print(np.mean(test.pore_permeability[l6:l7]))
print(np.mean(test.pore_permeability[l7:]))
print(" ")
print("Vsh SP")
print(np.mean(test.VshaleSP[0:l2]))
print(np.mean(test.VshaleSP[l2:l3]))
print(np.mean(test.VshaleSP[l3:l4]))
print(np.mean(test.VshaleSP[l4:l5]))
print(np.mean(test.VshaleSP[l5:l6]))
print(np.mean(test.VshaleSP[l6:l7]))
print(np.mean(test.VshaleSP[l7:]))
print(" ")
print("Vsh GR")
print(np.mean(test.VshaleGR[0:l2]))
print(np.mean(test.VshaleGR[l2:l3]))
print(np.mean(test.VshaleGR[l3:l4]))
print(np.mean(test.VshaleGR[l4:l5]))
print(np.mean(test.VshaleGR[l5:l6]))
print(np.mean(test.VshaleGR[l6:l7]))
print(np.mean(test.VshaleGR[l7:]))
print(" ")
print("Sw SP Indonesia")
print(np.mean(test.Indonesia_Sw[0:l2]))
print(np.mean(test.Indonesia_Sw[l2:l3]))
print(np.mean(test.Indonesia_Sw[l3:l4]))
print(np.mean(test.Indonesia_Sw[l4:l5]))
print(np.mean(test.Indonesia_Sw[l5:l6]))
print(np.mean(test.Indonesia_Sw[l6:l7]))
print(np.mean(test.Indonesia_Sw[l7:]))
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


a = test.getPorosityCut()
b = test.getPermeabilityCut()
e = test.getShaleCut(test.VshaleGR)
#
# d = test.Depth*a**c
#
# test.plotSingle(a*e*b, "Cutoff Plot", "(-)")

#
# test.plotSingle(test.VshaleGR, "Shale Volume GR")
# test.plotSingle(test.VshaleSP, "Shale Volume SP")

test_excel = pd.DataFrame(test.effective_porosity)
test_excel.columns = ["Depth", "Density", "Neutron", "Effective"]
sheet_name = "test_data.xls"
test_excel.to_excel(sheet_name, index=False)