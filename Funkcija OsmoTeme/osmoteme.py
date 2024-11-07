import numpy as np
#funckije za pretvaranje u homogenu i afinu reprezentaciju tacke
def homogena(tacka):
    return [tacka[0], tacka[1], 1]

def afina(tacka):
    return [tacka[0]/tacka[2], tacka[1]/tacka[2]]

def osmoteme(tacke):

    #potrebne su nam homogene tacke da bismo nasli beskonacne tacke
    t5 = homogena(tacke[0])
    t6 = homogena(tacke[1])
    t7 = homogena(tacke[2])
    t8 = homogena(tacke[3])
    t1 = homogena(tacke[4])
    t2 = homogena(tacke[5])
    t3 = homogena(tacke[6])

    #vektorski proizvodi za racunanje beskonacnih tacaka
    p12 = np.cross(t1, t2)
    p78 = np.cross(t7, t8)
    Y = np.cross(p12, p78)

    p37 = np.cross(t3, t7)
    p26 = np.cross(t2, t6)
    X = np.cross(p37, p26)

    #ukrstanje da se nadje t4 preko beskonacne prave i datih tacaka koje dele x i y sa nepoznatom
    Q = np.cross(t3, Y)
    P = np.cross(t8, X)


    t4 = np.cross(P, Q)
    t4 = afina(t4)

    return t4
