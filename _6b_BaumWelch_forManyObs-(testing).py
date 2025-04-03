import numpy as np
from copy import deepcopy
from random import random
from math import log

from _4a_doPrzodu_forward import forward
from _4b_doTylu_backward import backward
from _6a_BaumWelch_forOneObs import printParams, BaumWelchOneObs

def BaumWelchManyObs(obsLista, priorPi, przejsciaM, emisjaE, steps = 10, shouldPrint=False):
    def identity( i, j ):
        if i == j:
            return 1
        else:
            return 0

    pi = deepcopy(priorPi)
    przejscia = deepcopy(przejsciaM)
    emisja = deepcopy(emisjaE)
    bestPi, bestM, bestE = pi, przejscia, emisja
    R = len( obsLista)
    T = min( [ len( obs) for obs in obsLista] )
    N = len(przejsciaM)
    NO = len( emisjaE[0] )
    step = 0
    while step < steps:
        if shouldPrint:
            print('### Krok', step,'###')
            printParams( pi, przejscia, emisja)

        etaRazem, xiRazem, eta, xi = [], [], [], []
        for nrObs in range(len(obsLista)):
            #if shouldPrint:
            #    print('\nobserwacja =',obsLista[nrObs])
            piCurr, przejsciaCurr, emisjaCurr, eta, xi = BaumWelchOneObs(obsLista[nrObs], pi, przejscia, emisja, steps = 1)
            etaRazem.append( eta)
            xiRazem.append( xi)

        pi2 = [ sum( etaRazem[r][1][i] for r in range(R) ) for i in range(N) ]     # teoretycznie miało by być tak, ale następny wiersz działa lepiej
        #pi2 = [ sum( (etaRazem[r][0][i]+etaRazem[r][1][i])/2 for r in range(R) ) for i in range(N) ]
        #normalizacja pi:
        pi2 = [ p/np.sum(pi2) for p in pi2 ]
        przejscia2 = np.array( [[
            sum( sum( xiRazem[r][t][(i,j)] for t in range(1,T) ) for r in range(R)      # range(T-1) -> range(1,T)
                 )/ sum( sum( etaRazem[r][t][i] for t in range(1,T) ) for r in range(R) # range(T-1) -> range(1,T)
                         ) for j in range(N) ] for i in range(N) ] )
        emisja2 = np.array( [[
            sum( sum( identity( o, obsLista[r][t-1] ) * etaRazem[r][t][i] for t in range(1,T+1) ) for r in range(R)   #range(T)-> range(1,T+1), obsLista[r][t]->obsLista[r][t-11]
                 ) / sum( sum( etaRazem[r][t][i] for t in range(1,T+1) ) for r in range(R)                  #range(T)-> range(1,T+1)
                          ) for o in range(NO) ] for i in range(N) ] )
        

        diff, suma1, suma2 = 0, 0, 0
        for obs in obsLista:
            alfa2, prObs2 = forward( obs, pi2, przejscia2, emisja2 )
            alfa1, prObs1 = forward( obs, pi, przejscia, emisja )
            suma1 += log(prObs1)
            suma2 += log(prObs2)
            if abs( log(prObs2) - log(prObs1)) > diff:
                diff = abs( log(prObs2) - log(prObs1))

        if step == 0:
            bestSum = suma1
        if suma1 > suma2:
            if suma1 > bestSum:
                bestSum, bestPi, bestM, bestE = suma1, pi, przejscia, emisja
                print('step =',step,',\t suma =',bestSum)
        if shouldPrint:
            print('DIFF = ', diff*len(obsLista)/suma2 )
        if diff*len(obsLista)/abs(suma2) < 1e-17:   # kiedy względne ulepszenie mniejsze od 1e-7
            break

        '''
        # alternatywny wariant warunku zbieżności - czasem prościej dostosować parametry do zagadnienia
        diffPr = max( abs(przejscia[i][j] - przejscia2[i][j])
                        for j in range(N) for i in range(N) )
        diffEm = max( abs(emisja[i][j] - emisja2[i][j])
                        for j in range(NO) for i in range(N) )
        diffPi = max( abs(pi[i] - pi2[i])
                        for i in range(N) )
        if shouldPrint:
            print('DIFF = ',[ round(d,7) for d in [diffPi,diffPr,diffEm] ],'\t\t pi =', [ round(p,4) for p in pi2 ] )
        
        if max( (diffPr,diffPr,diffEm) ) < 0.0000001:
            break
        '''
        
        pi, przejscia, emisja = pi2, przejscia2, emisja2

        step += 1
        if step == 1:
            print('Maks.',steps,'iteracji: ')#,end='\t')
        if step % 100 == 0 and step > 0:
            #print( step, end='\t' )
            print( step,'\t', suma2 )
    print('\nBaumWelchManyObs: \n')
    print( 'iteracji =',step)
    printParams( pi, przejscia, emisja)
    print('średnia prawdop. obserwacji z sekwencji obserwacyjnej względem znaleźionych parametrów modelu = ', suma2/len(obsLista))

    print()                   
    printParams( bestPi, bestM, bestE )
    diff, suma = 0, 0
    for obs in obsLista:
        alfa1, prObs1 = forward( obs, bestPi, bestM, bestE )
        suma += log( prObs1 )
    print('suma =', suma, 'średnia =',suma/len(obsLista))
    return pi, przejscia, emisja, suma/len(obsLista)



if __name__ == '__main__':
    
    prior0 = [0.3, 0.4, 0.3]
    przejscia0 = np.array( [
        [ 0.34,  0.33,  0.33 ],
        [ 0.34,  0.33,  0.33 ],
        [ 0.34,  0.33,  0.33 ]
        ] )
    emisja0 = np.array( [
        [ 0.34,  0.33,  0.33 ],
        [ 0.34,  0.33,  0.33 ],
        [ 0.34,  0.33,  0.33 ]
        ] )
    
    #obsLista = [[1, 1, 1, 2, 1, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 2, 1, 2, 0, 1, 2, 2], [0, 2, 0, 2, 1, 2, 2, 2, 2, 2, 0], [0, 2, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1], [0, 0, 0, 0, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 0, 2, 0, 0, 2, 1, 2, 2, 2, 0, 2, 2, 1], [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0], [0, 2, 0, 0, 2, 2, 2, 2, 0, 0, 1, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2], [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1], [0, 0, 0, 0, 1, 2, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 2, 2, 2, 0, 2, 2, 1, 1, 1, 1], [0, 2, 0, 1, 0, 2, 2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 2], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 1], [0, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2], [0, 0, 0, 0, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 0, 2], [2, 0, 0, 0, 0, 1, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 0, 1], [1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 1, 2, 1, 0, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 2], [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0]]
    #obsLista = [[2, 7, 7, 13, 9, 7, 9, 6, 7, 3, 6, 8, 12, 14, 13, 6, 7, 0, 0, 12], [2, 12, 5, 7, 12, 11, 4, 2, 13, 7, 11, 12, 8, 14, 9, 10, 12], [7, 13, 9, 8, 5, 14, 0, 11, 4, 13, 2, 12, 6], [9, 0, 7, 14, 12, 13, 10, 3, 2, 2, 1, 2, 1], [1, 10, 13, 9, 1, 9, 14, 0, 6, 1, 5, 7, 7, 3, 2, 10], [9, 14, 13, 8, 5, 12, 9, 8, 7, 4, 9, 7, 5, 14, 10, 8, 3, 4, 6, 8, 13], [4, 13, 2, 4, 10, 12, 9, 13, 7, 9, 9, 1, 2, 13, 0], [0, 10, 0, 10, 4, 12, 6, 8, 8, 3, 10, 13, 14, 14], [12, 13, 6, 2, 4, 6, 6, 2, 4, 6, 0, 13, 0, 0, 0], [4, 10, 4, 0, 7, 4, 6, 0, 9, 4, 9, 6, 12, 7, 7, 9], [9, 1, 9, 5, 2, 8, 5, 4, 13, 5, 5, 4, 14, 12, 5, 4, 1, 3], [13, 12, 13, 11, 8, 11, 2, 1, 8, 8, 13, 12, 2, 6, 12], [0, 13, 8, 1, 13, 1, 4, 6, 14, 6, 3, 13, 7, 10, 9, 6, 12], [0, 13, 7, 11, 14, 13, 10, 9, 0, 9, 5, 0, 8], [10, 7, 13, 11, 2, 6, 10, 1, 4, 8, 8, 5, 14, 2], [12, 6, 6, 7, 6, 5, 7, 11, 3, 10, 8, 11, 11, 1, 5, 11, 3, 9], [11, 14, 9, 8, 13, 7, 8, 12, 10, 8, 1, 7, 3, 14, 6, 5, 9, 6, 3, 14, 8], [0, 6, 4, 0, 5, 7, 10, 2, 14, 0, 5, 11, 14, 8, 10, 9]]
    obsLista = [[4, 0, 2, 1, 0, 0, 1, 3, 3, 2, 1, 0, 2, 3, 2, 1, 1, 1, 2, 0], [4, 2, 2, 4, 2, 1, 1, 4, 1, 2, 0, 2, 1, 1, 4, 3, 1, 2, 2], [4, 0, 0, 1, 4, 4, 3, 3, 0, 2, 3, 3, 2, 1, 0], [4, 3, 2, 1, 2, 2, 0, 2, 2, 2, 0, 1, 1, 2, 4], [4, 2, 2, 1, 1, 1, 1, 2, 4, 1, 3, 2, 0, 4, 1, 3, 0], [3, 2, 3, 2, 1, 0, 0, 2, 3, 3, 3, 2, 1, 0, 1, 0, 2, 2, 3, 4, 4], [3, 3, 2, 0, 1, 1, 2, 4, 0, 0, 0, 1, 4, 1, 1], [0, 4, 3, 3, 1, 1, 4, 3, 2, 2, 3, 2, 2], [4, 2, 1, 3, 4, 1, 2, 2, 0, 0, 1, 2, 1], [4, 1, 0, 0, 3, 2, 4, 3, 4, 2, 0, 1, 3, 2, 4, 3], [1, 2, 0, 3, 3, 0, 3, 3, 0, 2, 0, 3, 0, 2, 3, 2, 2, 1], [2, 1, 1, 3, 1, 2, 2, 4, 2, 0, 2, 3, 2, 2, 4, 0, 0, 3, 3], [0, 3, 2, 4, 4, 2, 2, 3, 2, 2, 3, 0, 1, 1, 3], [1, 2, 2, 0, 4, 2, 2, 1, 2, 2, 4, 2, 2, 0, 3], [1, 3, 1, 2, 0, 2, 2, 4, 2, 3, 3, 0, 4, 3, 2, 2], [1, 3, 4, 4, 2, 1, 3, 0, 2, 2, 2, 4, 1], [4, 0, 2, 1, 2, 3, 2, 0, 0, 2, 1, 3, 2, 2, 0, 3], [2, 4, 2, 3, 3, 3, 2, 2, 3, 3, 4, 4, 3]]
    obsLista = [[2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1], [2, 2, 0, 1, 0, 0, 0, 0, 2, 2, 2, 0, 2, 0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 0, 2, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 2, 0, 1], [1, 0, 0, 0, 0, 1, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 2, 0], [0, 1, 1, 1, 2, 0, 0, 1, 2, 2, 2, 0, 0, 0, 1, 0, 1, 1], [1, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 1, 1, 2, 0], [2, 0, 1, 1, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 2, 2, 0, 1, 1, 2, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 0], [0, 1, 2, 0, 1, 1, 1, 1, 1, 2, 2, 1, 0, 1, 0, 2, 2, 0, 1], [0, 0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 0, 0, 1], [2, 0, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 0, 1, 1], [2, 2, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0], [1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    N, NO = 3, max( [ o for obs in obsLista for o in obs  ])+1
    prior0 = [1/N for i in range(N) ]
    przejscia0 = np.zeros( (N,N) )
    emisja0 = np.zeros( (N,NO) )
    przejscia0 = np.array( [[ 1/N for j in range(N) ] for i in range(N) ] )
    emisja0 = np.array( [[ 1/NO for j in range(NO) ] for i in range(N) ] )

    prior0 = [0.1, 0.6, 0.3]
    #print('\nobserwacyjna sekwencja: ')
    #for o in obsLista[:]:
    #    print(o)
    print()
    BaumWelchManyObs(obsLista[:], prior0, przejscia0, emisja0, steps = 300, shouldPrint=False)

