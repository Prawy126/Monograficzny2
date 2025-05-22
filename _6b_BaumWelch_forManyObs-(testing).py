import numpy as np
from copy import deepcopy
from random import random
from math import log, exp

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
            alfa2, prObs2 = forward( obs, przejscia2, emisja2, pi2 )
            alfa1, prObs1 = forward( obs, przejscia, emisja, pi )
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
        alfa1, prObs1 = forward( obs, bestM, bestE, bestPi )
        suma += log( prObs1 )
    print('suma =', suma, 'średnia =',suma/len(obsLista))
    return pi, przejscia, emisja, suma/len(obsLista)



if __name__ == '__main__':
    '''
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
    '''
    
    #obsLista = [[1, 1, 1, 2, 1, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 2, 1, 2, 0, 1, 2, 2], [0, 2, 0, 2, 1, 2, 2, 2, 2, 2, 0], [0, 2, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1], [0, 0, 0, 0, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 0, 2, 0, 0, 2, 1, 2, 2, 2, 0, 2, 2, 1], [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0], [0, 2, 0, 0, 2, 2, 2, 2, 0, 0, 1, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2], [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1], [0, 0, 0, 0, 1, 2, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 2, 2, 2, 0, 2, 2, 1, 1, 1, 1], [0, 2, 0, 1, 0, 2, 2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 2], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 1], [0, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2], [0, 0, 0, 0, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 0, 2], [2, 0, 0, 0, 0, 1, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 0, 1], [1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 1, 2, 1, 0, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 2], [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0]]
    #obsLista = [[2, 7, 7, 13, 9, 7, 9, 6, 7, 3, 6, 8, 12, 14, 13, 6, 7, 0, 0, 12], [2, 12, 5, 7, 12, 11, 4, 2, 13, 7, 11, 12, 8, 14, 9, 10, 12], [7, 13, 9, 8, 5, 14, 0, 11, 4, 13, 2, 12, 6], [9, 0, 7, 14, 12, 13, 10, 3, 2, 2, 1, 2, 1], [1, 10, 13, 9, 1, 9, 14, 0, 6, 1, 5, 7, 7, 3, 2, 10], [9, 14, 13, 8, 5, 12, 9, 8, 7, 4, 9, 7, 5, 14, 10, 8, 3, 4, 6, 8, 13], [4, 13, 2, 4, 10, 12, 9, 13, 7, 9, 9, 1, 2, 13, 0], [0, 10, 0, 10, 4, 12, 6, 8, 8, 3, 10, 13, 14, 14], [12, 13, 6, 2, 4, 6, 6, 2, 4, 6, 0, 13, 0, 0, 0], [4, 10, 4, 0, 7, 4, 6, 0, 9, 4, 9, 6, 12, 7, 7, 9], [9, 1, 9, 5, 2, 8, 5, 4, 13, 5, 5, 4, 14, 12, 5, 4, 1, 3], [13, 12, 13, 11, 8, 11, 2, 1, 8, 8, 13, 12, 2, 6, 12], [0, 13, 8, 1, 13, 1, 4, 6, 14, 6, 3, 13, 7, 10, 9, 6, 12], [0, 13, 7, 11, 14, 13, 10, 9, 0, 9, 5, 0, 8], [10, 7, 13, 11, 2, 6, 10, 1, 4, 8, 8, 5, 14, 2], [12, 6, 6, 7, 6, 5, 7, 11, 3, 10, 8, 11, 11, 1, 5, 11, 3, 9], [11, 14, 9, 8, 13, 7, 8, 12, 10, 8, 1, 7, 3, 14, 6, 5, 9, 6, 3, 14, 8], [0, 6, 4, 0, 5, 7, 10, 2, 14, 0, 5, 11, 14, 8, 10, 9]]
    #obsLista = [[4, 0, 2, 1, 0, 0, 1, 3, 3, 2, 1, 0, 2, 3, 2, 1, 1, 1, 2, 0], [4, 2, 2, 4, 2, 1, 1, 4, 1, 2, 0, 2, 1, 1, 4, 3, 1, 2, 2], [4, 0, 0, 1, 4, 4, 3, 3, 0, 2, 3, 3, 2, 1, 0], [4, 3, 2, 1, 2, 2, 0, 2, 2, 2, 0, 1, 1, 2, 4], [4, 2, 2, 1, 1, 1, 1, 2, 4, 1, 3, 2, 0, 4, 1, 3, 0], [3, 2, 3, 2, 1, 0, 0, 2, 3, 3, 3, 2, 1, 0, 1, 0, 2, 2, 3, 4, 4], [3, 3, 2, 0, 1, 1, 2, 4, 0, 0, 0, 1, 4, 1, 1], [0, 4, 3, 3, 1, 1, 4, 3, 2, 2, 3, 2, 2], [4, 2, 1, 3, 4, 1, 2, 2, 0, 0, 1, 2, 1], [4, 1, 0, 0, 3, 2, 4, 3, 4, 2, 0, 1, 3, 2, 4, 3], [1, 2, 0, 3, 3, 0, 3, 3, 0, 2, 0, 3, 0, 2, 3, 2, 2, 1], [2, 1, 1, 3, 1, 2, 2, 4, 2, 0, 2, 3, 2, 2, 4, 0, 0, 3, 3], [0, 3, 2, 4, 4, 2, 2, 3, 2, 2, 3, 0, 1, 1, 3], [1, 2, 2, 0, 4, 2, 2, 1, 2, 2, 4, 2, 2, 0, 3], [1, 3, 1, 2, 0, 2, 2, 4, 2, 3, 3, 0, 4, 3, 2, 2], [1, 3, 4, 4, 2, 1, 3, 0, 2, 2, 2, 4, 1], [4, 0, 2, 1, 2, 3, 2, 0, 0, 2, 1, 3, 2, 2, 0, 3], [2, 4, 2, 3, 3, 3, 2, 2, 3, 3, 4, 4, 3]]
    #obsLista = [[2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1], [2, 2, 0, 1, 0, 0, 0, 0, 2, 2, 2, 0, 2, 0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 0, 2, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 1, 1, 2, 0, 1], [1, 0, 0, 0, 0, 1, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 2, 0], [0, 1, 1, 1, 2, 0, 0, 1, 2, 2, 2, 0, 0, 0, 1, 0, 1, 1], [1, 1, 2, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 1, 1, 2, 0], [2, 0, 1, 1, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 2, 2, 0, 1, 1, 2, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 0], [0, 1, 2, 0, 1, 1, 1, 1, 1, 2, 2, 1, 0, 1, 0, 2, 2, 0, 1], [0, 0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 0, 0, 1], [2, 0, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 0, 1, 1], [2, 2, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 2, 0, 0, 0, 0, 0, 0], [1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    obsLista = [[2, 1, 3, 1, 2, 0, 1, 4, 1, 1, 0, 1, 1, 1, 1, 4, 1, 0],[3, 4, 2, 0, 1, 1, 1, 4, 0, 0, 0, 0, 3, 2, 1, 3, 2, 4, 3],[2, 0, 3, 0, 2, 2, 0, 4, 1, 3, 1, 0, 1, 2, 0, 2, 1, 0],[0, 0, 4, 3, 1, 2, 1, 3, 1, 0, 0, 2, 1, 2, 4, 4],[1, 3, 2, 2, 1, 2, 1, 1, 1, 1, 2, 3, 3],[2, 2, 4, 1, 3, 2, 1, 4, 0, 1, 2, 0, 1, 0, 0, 4],[1, 1, 1, 1, 1, 2, 1, 1, 0, 2, 0, 3, 4, 1, 1, 3, 4, 1, 0],[0, 0, 3, 0, 1, 1, 2, 4, 1, 0, 3, 1, 0, 1, 1, 3],[2, 1, 3, 0, 0, 0, 2, 4, 3, 1, 1, 0, 1],[0, 0, 2, 1, 3, 0, 0, 3, 2, 2, 3, 0, 4, 0, 1, 4, 3, 0, 1, 1],[2, 0, 1, 1, 3, 0, 0, 3, 0, 0, 0, 1, 2, 0, 1, 3, 3, 2],[1, 0, 4, 1, 1, 1, 0, 2, 3, 0, 0, 2, 1, 2, 2],[2, 2, 3, 0, 1, 0, 0, 4, 1, 2, 1, 0, 2],[1, 0, 1, 2, 1, 0, 0, 2, 3, 0, 0, 3, 3, 0, 1, 4, 2, 3, 3, 3],[2, 3, 2, 3, 1, 0, 1, 4, 0, 0, 0, 3, 1, 1, 1, 2, 1, 4, 3],[0, 1, 3, 1, 2, 1, 0, 3, 1, 0, 0, 1, 1, 3, 0, 3, 1],[0, 1, 3, 1, 1, 0, 1, 4, 1, 2, 0, 3],[1, 1, 4, 1, 0, 0, 0, 4, 0, 3, 2, 0, 1, 2, 1, 3, 1, 1, 3, 1],[3, 1, 2, 1, 1, 0, 0, 4, 2, 0, 0, 2, 1],[1, 1, 2, 2, 2, 1, 1, 1, 3, 1, 2, 0, 0, 1, 2, 4, 1, 2, 2],[0, 1, 4, 0, 1, 0, 2, 4, 1, 1, 0, 0, 3, 0, 0, 2, 1, 1, 3],[1, 1, 2, 2, 1, 2, 0, 3, 0, 0, 0, 1],[1, 2, 3, 2, 1, 1, 0, 1, 1, 0, 0, 0, 0, 2]]

    N, NO = 3, len(set(  [ o for obs in obsLista for o in obs  ] ))   #max( [ o for obs in obsLista for o in obs  ])+1
    prior0 = [1/N for i in range(N) ]
    przejscia0 = np.zeros( (N,N) )
    przejscia0 = np.array( [[ 1/N for j in range(N) ] for i in range(N) ] )
    emisja0 = np.zeros( (N,NO) )
    emisja0 = np.array( [[ 1/NO for j in range(NO) ] for i in range(N) ] )

    prior0 = [0.1, 0.6, 0.3]
    #print('\nobserwacyjna sekwencja: ')
    #for o in obsLista[:]:
    #    print(o)
    print()
    pi, przejscia, emisja,sm = BaumWelchManyObs(obsLista[:], prior0, przejscia0, emisja0, steps = 300, shouldPrint=False)

    from _4d_viterbi import viterbi
    trajektorie = []
    obs_probablities = []
    for obs in obsLista:
        trajekt, pr, _ = viterbi( obs, przejscia, emisja )
        trajektorie.append( trajekt)
        obs_probablities.append( pr[-1][-1] )
        print( f'obs = {obs}, trajekt, pr = {trajekt, pr[-1][-1]}')

    stany_obs = sorted(list(  set(  [ o for obs in obsLista for o in obs  ] ) ))
    minTime = min([len(obs) for obs in obsLista]) 
    '''
    election = [ { sto: 0 for sto in stany_obs } for i in range( minTime )]
    for it,trajekt in enumerate(trajektorie):
        for week in range(minTime):
            #print( f'it, trajekt[week], week = {it, trajekt[week], week}' )
            election[week][ trajekt[week] ] += log( obs_probablities[it] )

    def normalizedProbList( probs ):
        if sum([p for p in probs ]) != 0:
            return [ p/sum([p for p in probs ]) for p in probs ]
        else:
            return probs
        
    for ir,row in enumerate(election):
        for sto in stany_obs:
            if election[ir][sto] != 0:
                election[ir][sto] = exp( election[ir][sto] )
        probs = list(election[ir].values())
        probs = normalizedProbList( probs )
        for sto in stany_obs:
            election[ir][sto] = probs[sto]

    for iw,week in enumerate(election):
        print( f'week {iw}: {week}' )
    '''
    ukryta_trajektoria = [ [] for i in range(minTime) ]
    for tr in trajektorie:
        for i in range(minTime):
            ukryta_trajektoria[i].append( tr[i] )
    #print( ukryta_trajektoria )
    laczna_ukryta_trajektoria = []
    for i in range(minTime):
        freq = [ ukryta_trajektoria[i].count(sto) for sto in stany_obs ]
        laczna_ukryta_trajektoria.append( np.argmax(freq) )
        print( freq, laczna_ukryta_trajektoria )
    print( f'laczna_ukryta_trajektoria = {laczna_ukryta_trajektoria}' )

    # jeśli mamy kandydata na ścieżkę ukrytych stanów (i również dane modelu prior,przejscia,emisja),
    # to możemy obliczyć pr-wo wystąpienia każdej z obserwacji ( w zależności od ukrytych stanów):
    # nestety, taki sposób pozwala porównywać macierzy emisji, ale nie macierzy przejścia ukrytych stanów
    Plista = []
    for obs in obsLista:
        P_obs = 0
        for i in range(minTime):
            P_obs += log( emisja[ laczna_ukryta_trajektoria[i], obs[i] ] )
        P_obs = exp( P_obs)
        Plista.append( P_obs )
        print( obs, '\t\t', P_obs)
        
    P_emm = 1
    for P in Plista:
        P_emm += log( P )
    #P_all = exp( P_all )
    print( f'ocena odpowiedności emisji =  {P_emm}' )

    P_ukr = 1
    for ist, stan in enumerate(laczna_ukryta_trajektoria):
        if ist == 0:
            continue
        P_ukr += log( przejscia[ laczna_ukryta_trajektoria[ist-1], laczna_ukryta_trajektoria[i] ] )
    print( f'ocena odpowiedności macierzy prześcia stanów ukrytych =  {P_ukr}' )

#[0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1]
#[1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 1, 2]
#[2, 0, 1, 2, 0, 2, 0, 1, 2, 0, 2, 0]
