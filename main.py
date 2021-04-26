from Interperter.loginterperter import Logview

test = Logview(r"FEP.xls", 'DE4 Ascii')
#test.Plot()
#test.calculateVshale()
#test.plotSingle(test.Vshale, name="V shale (GR)")
#test.getShaleLine()
#test.getSandstoneLine()
test.plotPorosityVsDensity()
#test.estimateLith()
#test.shaleVolumeFormula()
#test.plotScatter()
#test.shaleVolumeFormula()
#test

#0.03 r_w