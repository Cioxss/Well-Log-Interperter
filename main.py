from Interperter.loginterperter import Logview
import numpy as np
import matplotlib.pyplot as plt
test = Logview(r"FEP.xls", 'DE1 Ascii')

test.plotPorosityVsDensity()

test.getArchie()
test.calculateVshaleGR()
test.getIndonesia("SP")
test.getTimur("Archie")
test.getTimur("Indonesia")
# test.getKroom()
test.getPorePermeability()

test.plotDouble(test.NEUT*100, test.effective_porosity[:,3])

test.plotSingle(test.effective_porosity[:,3], "Effective Porosity")
#
# test.plotSingle(test.pore_permeability, "Permeability")
#
# test.plotSingle(test.VshaleGR, "Shale Volume GR")
# test.plotSingle(test.VshaleSP, "Shale Volume SP")
# test.plotSingle(test.Archie_Sw, "Archie S_w")
# test.plotSingle(test.Indonesia_Sw, "Indonesia S_w (SP)")
# test.getIndonesia("GR")
# test.plotSingle(test.Indonesia_Sw, "Indonesia S_w (GR)")