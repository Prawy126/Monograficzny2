import numpy as np
from copy import deepcopy
from random import random

from _4a_doPrzodu_forward import forward
from _4b_doTylu_backward import backward
from _4d_viterbi import viterbi

def printParams( priorPi, przejsciaM, emisjaE ):
    print('początkowe prawdopodobieństwa:\t', '{' \
          + ', '.join( f'{round(x,4):.4f}'  for x in priorPi ) + '}')
    print('pr.przejścia:')
    for i in range(len(przejsciaM)):
        print( '' + ', '.join( f'{round(x,4):.4f}'
                               for x in przejsciaM[i] ) + '')

    print('pr.emisji:')
    for i in range(len(przejsciaM)):
        print( '' + ', '.join( f'{round(x,4):.4f}'
                               for x in emisjaE[i] ) + '' )        
    print()
    return

def normalizedProbOne( alfa ):
    if sum([alfa[k] for k in alfa.keys()]) == 0:
        return alfa, 0
    else:
        alfa_denom = sum([alfa[k] for k in alfa.keys()])
        return { k:(alfa[k]/alfa_denom)
                 for k in alfa.keys() }, alfa_denom

def normalizedProbMany( alfa ):
    alfa2 = [{}]
    alfa_denom = [{}]
    for i,alf in enumerate(alfa):
        if i > 0:
            a, d = normalizedProbOne( alf)
            alfa2.append( a)
            if d > 0:
                alfa_denom.append( d)
            else:
                alfa_denom.append( 1)
    return alfa2, alfa_denom

def evaluateEta(obs, alfa, beta, N):
    T = len(obs)
    eta = [{} for t in range(T+1) ]     # T -> T+1, bo teraz [0] nie używa się        
        
    for t in range(1,T+1):          # range(T-1) -> range(1,T), czas [0] nie używa się
        mianownik = sum( alfa[t][zI] * beta[t][zI] for zI in range(N) )
        if mianownik == 0:
            print('mianownik ETA (dla t='+str(t)+') = ', mianownik )
        for zI in range(N):
            if mianownik > 0:
                eta[t][zI] = alfa[t][zI] * beta[t][zI] / mianownik
            else:
                eta[t][zI] = 0
    return eta

def evaluateXi(obs, przejscia, emisja, alfa, beta, N):
    T = len(obs)
    O = [''] + obs

    xi = [{} for t in range(T) ]      # T-1 -> T, bo teraz [0] nie używa się 
    for t in range(1,T):                # range(T-1) -> range(1,T)
        mianownik = sum( [ sum( [
                    alfa[t][zI] \
                    * przejscia[zI][zJ] \
                    * beta[t+1][zJ] \
                    * emisja[zJ][
                        O[t+1] ]
                    for zI in range(N) ] ) for zJ in range(N) ] )
        if mianownik == 0:
            print('mianownik XI (dla t='+str(t)+') = ', mianownik )
        for zt in range(N):
            for ztplus1 in range(N):
                if mianownik > 0:
                    xi[t][(zt,ztplus1)] = alfa[t][zt] * przejscia[zt][ztplus1] \
                    * beta[t+1][ztplus1] * emisja[ztplus1][ O[t+1]] \
                    / mianownik
                else:
                    xi[t][(zt,ztplus1)] = 0
    return xi

def BaumWelchOneObs(obs, priorPi, przejsciaM, emisjaE, steps = 10, shouldPrint=False):
    if shouldPrint:
        print('\n######## Baum-Welch algorithm ##########\n')
    T = len(obs)
    N = len(przejsciaM)
    NO = len( emisjaE[0] )
    O = [''] + obs      # O[0]=='' nie używa się
    
    pi = deepcopy(priorPi)
    przejscia = deepcopy(przejsciaM)
    emisja = deepcopy(emisjaE)
    
    def identity( i, j ):
        if i == j:
            return 1
        else:
            return 0

    step = 0
    #eta = [{} for t in range(T) ]
    #xi = [{} for t in range(T-1) ]
    while step < steps:
        if shouldPrint:
            print('### Krok', step,'###')
            printParams( pi, przejscia, emisja)

        alfa, prObsAlfa = forward( obs, przejscia, emisja, pi )
        alfa, alfa_denom = normalizedProbMany( alfa )
        beta, prObsBeta = backward( obs, przejscia, emisja, pi )
        betaNorm = [{}] + [ { zI:(beta[t][zI] / alfa_denom[t]) for zI in beta[t].keys() } 
                 for t in range(1,len(beta)) ]            #range(len(beta)) -> range(1,len(beta))
        beta = betaNorm
        eta = evaluateEta(obs, alfa, beta, N)
        xi = evaluateXi(obs, przejscia, emisja, alfa, beta, N)

        '''
        if shouldPrint:
            print('alfa = ', '\n'.join( [ str(d) for d in alfa[1:] ] ) )
            print('beta = ', '\n'.join( [ str(d) for d in beta[1:] ] ) )
            print('eta = ', '\n'.join( [ str(d) for d in eta[1:] ] ) )
            print('xi = ', '\n'.join( [ str(d) for d in xi[1:] ] ) )
        '''

        pi2 = [ eta[1][i] for i in range(N) ]    # teoretycznie miało by być tak, ale czasem następny wiersz działa lepiej
        #pi2 = [ (eta[1][i]+eta[2][i])/2 for i in range(N) ]
        przejscia2 = np.array( [[ sum( xi[t][(i,j)] for t in range(1,T)     # range(T-1) -> range(1,T)
                                      ) / sum( eta[t][i] for t in range(1,T) )   # range(T-1) -> range(1,T)
                      for j in range(N) ] for i in range(N) ] )
        emisja2 = np.array( [[ sum( identity( o, O[t] ) * eta[t][i] for t in range(1,T+1)  # range(T) -> range(1,T+1)
                                   ) / sum( eta[t][i] for t in range(1,T+1) )   # range(T) -> range(1,T+1)z
                      for o in range(NO) ] for i in range(N) ] )
        
        if steps > 1:
            alfa2, prObs2 = forward( obs, przejscia2, emisja2, pi2)
            alfa1, prObs1 = forward( obs, przejscia, emisja, pi2 )
            diff = abs( prObs2 - prObs1)/prObs2
            if shouldPrint:                
                print('DIFF = ',diff,'\t\t prawd.obs. = ', prObs2) 
            if diff < 1e-17:
                break        

        pi, przejscia, emisja = pi2, przejscia2, emisja2
        step += 1
        if step == 1 and steps > 1:
            print('Maks.',steps,'iteracji: ')
        if step % 100 == 0 and step > 0:
            print( step,'\t', prObs2 )
            #printParams( pi, przejscia, emisja)
    if step > 1:            # tzn. funkcja wykonuje się autonomnie, a nie jako podrzędna z funkcji "BaumWelchManyObs"
        print('\nFunkcja "BaumWelchOneObs": \n')
        print( 'iteracji =',step)
        printParams( pi, przejscia, emisja)
        print('prawdop. obserwacji względem znaleźionych parametrów modelu = ', prObs2 )
    return pi, przejscia, emisja, eta, xi



if __name__ == '__main__':

    '''    
    obs = [0,1,2]
    prior = [0.34, 0.33, 0.33] 
    przejscia = np.array( [
        [ 0.6, 0.2,  0.2 ],
        [ 0.1,  0.8, 0.1 ],
        [ 0.1, 0.1, 0.8 ]
        ] )
    emisja = np.array( [
        [ 0.8,  0.1,  0.1 ],
        [ 0.1,  0.8,  0.1 ],
        [ 0.1,  0.1,  0.8 ]
        ] )
    '''
    
    #obs = [1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 1, 1, 2, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0, 2, 0, 2, 2, 2, 0]
    #obs = [2, 0, 1, 1, 0, 0, 2, 2, 2, 2, 1, 0, 0, 0, 0, 2, 2, 2, 1, 2, 2]
    #obs = [0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]
    #obs = [2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1]#, 0, 0, 1, 1]

    obs =  [1,1,1,2,2,2,1,1,0,1,2,2 ]
    
    N, NO = 3, max(obs)+1
    prior0 = [1/N for i in range(N) ]
    przejscia0 = np.zeros( (N,N) )
    emisja0 = np.zeros( (N,NO) )
    przejscia0 = np.array( [[ 1/N for j in range(N) ] for i in range(N) ] )
    emisja0 = np.array( [[ 1/NO for j in range(NO) ] for i in range(N) ] )

    #prior0=[0.4,0.2,0.4]

    #print('Wprowadzane wartości początkowe w metodę Bauma-Welcha: ')
    #printParams( prior0, przejscia0, emisja0)
    print('obs =',obs, ',  prior =',prior0)
    print()
    pi, prz, emi, eta, xi = BaumWelchOneObs(obs, prior0, przejscia0, emisja0,
                                            steps = 2000, shouldPrint=False)
    printParams( pi, prz, emi)
    most_likely_hiden_states, _, _ = viterbi(obs, prz, emi, priorPi=pi)
    print(f"most likely hiden states = {most_likely_hiden_states}")
    print("\n\n")

    prior0=[0.2,0.6,0.2]
    print('obs =',obs, ',  prior =',prior0)
    print()
    pi, prz, emi, eta, xi = BaumWelchOneObs(obs, prior0, przejscia0, emisja0,
                                            steps = 2000, shouldPrint=False)

    most_likely_hiden_states, _, _ = viterbi(obs, prz, emi, priorPi=pi)
    print(f"most likely hiden states = {most_likely_hiden_states}")
    # można zauważyć, że przy nieco różnych 'priorPi' wynik działania (obliczone macierzy) może znacząco różnić się
    # ( choć prawdop. obserwacji względem znaleźionych parametrów modelu są rowne w obu przypadkach )
    
