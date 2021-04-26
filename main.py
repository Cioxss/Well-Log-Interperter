from Interperter.loginterperter import Logview

test = Logview(r"FEP.xls", 'DE1 Ascii')

test.calcMeanVshale()
#test.plotVshale()
test.plotPorosityVsDensity()
test.getArchie()
print(test.Archie_Sw)
test.getIndonesia("GR")


#0.03 r_w