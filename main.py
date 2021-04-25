from Interperter.loginterperter import Logview

test = Logview(r"FEP.xls", lb=600, rb=1500)
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