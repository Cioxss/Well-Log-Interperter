from Interperter.loginterperter import Logview

test = Logview(r"FEP.xls", 'DE1 Ascii')

test.plotPorosityVsDensity()
test.getArchie()
test.getIndonesia("SP")
test.getTimur("Archie")
print(test.permeability)
test.getTimur("Indonesia")
print(test.permeability)


#0.03 r_w