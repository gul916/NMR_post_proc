import topspin_inout as inout
import sys
import signalTopspin as sig
import signal_processing as sigP

fulldataPATH = sys.argv[1]
fullEcho = sys.argv[2]
nbEcho = sys.argv[3]
firstDec = sys.argv[4]

fullEcho = float(fullEcho)
fullEcho *= 1e-3
nbEcho = int(nbEcho)

if (int(firstDec)==1):
	firstDec = True
else:
	firstDec = False

print("fullEcho = ",fullEcho)
print("nbEcho = ",nbEcho)
print("firstDec = ",firstDec)

data, param = inout.load_data(fulldataPATH)
dw = float(1/(2*param['acqus']['SW_h']))
de = float(param['acqus']['DE'])
de *= 1e-6		# de is in microseconds in the parameter dictionary
TD = int(param['acqus']['TD'])
nbPt = int(TD / 2)
print("dw = ",dw)
print("de = ",de)
print("nbPt = ",nbPt)


processedSig = sig.Signal()
processedSig.setValues_user(firstDec,fullEcho,nbEcho)
processedSig.setValues_topspin(dw,nbPt,de)
processedSig.setData(data)

denData = sigP.signal_processing(processedSig)

input('\nPress enter key to exit') # have the graphs stay displayed even when launched from linux terminal

#inout.export_data(fulldataPATH, param, data)