import numpy as np
from copy import deepcopy
from random import random

from _4a_doPrzodu_forward import forward
from _4b_doTylu_backward import backward

def printParams( priorPi, przejsciaM, emisjaE ):
    print('początkowe prawdopodobieństwa:\t', '{' + ', '.join( f'{round(x,4):.4f}'  for x in priorPi ) + '}')
    print('pr.przejścia:' + '\t\t\t\t' + 'pr.emisji:')
    for i in range(len(przejsciaM)):
        print( '' + ', '.join( f'{round(x,4):.4f}'  for x in przejsciaM[i] ) + '', end='\t\t' )
        print( '' + ', '.join( f'{round(x,4):.4f}'  for x in emisjaE[i] ) + '' )
    print()
    return

def normalizedProbOne( alfa ):
    if sum([alfa[k] for k in alfa.keys()]) == 0:
        return alfa, 0
    else:
        alfa_denom = sum([alfa[k] for k in alfa.keys()])
        return { k:(alfa[k]/alfa_denom) for k in alfa.keys() }, alfa_denom

def normalizedProbMany( alfa ):
    alfa2 = []
    alfa_denom = []
    for alf in alfa:
        a, d = normalizedProbOne( alf)
        alfa2.append( a)
        if d > 0:
            alfa_denom.append( d)
        else:
            alfa_denom.append( 1)
    return alfa2, alfa_denom

def BaumWelchOneObs(obs, priorPi, przejsciaM, emisjaE, steps = 10, shouldPrint=False):
    if shouldPrint:
        print('\n######## Baum-Welch algorithm ##########\n')
    T = len(obs)
    N = len(przejsciaM)
    NO = len( emisjaE[0] )
    
    pi = deepcopy(priorPi)
    przejscia = deepcopy(przejsciaM)
    emisja = deepcopy(emisjaE)
    
    def unit( i, j ):
        if i == j:
            return 1
        else:
            return 0

    step = 0
    eta = [{} for t in range(T) ]
    xi = [{} for t in range(T-1) ]
    while step < steps:
        if shouldPrint:
            print('### Krok', step,'###')
            printParams( pi, przejscia, emisja)

        alfa, alfaAll = forward( obs, pi, przejscia, emisja )
        alfa, alfa_denom = normalizedProbMany( alfa )
        beta = backward( obs, pi, przejscia, emisja )
        beta = [ { zI:(beta[t][zI] / alfa_denom[t]) for zI in beta[t].keys() } 
                 for t in range(len(beta)) ]

        eta = [{} for t in range(T) ]            
            
        for t in range(T):
            mianownik = sum( alfa[t][zI] * beta[t][zI] for zI in range(N) )
            if mianownik == 0:
                print('mianownik ETA (dla t='+str(t)+') = ', mianownik )
            for zI in range(N):
                if mianownik > 0:
                    eta[t][zI] = alfa[t][zI] * beta[t][zI] / mianownik
                else:
                    eta[t][zI] = 0

        xi = [{} for t in range(T-1) ]
        for t in range(T-1):
            mianownik = sum( [ sum( [
                        alfa[t][zI] * przejscia[zI][zJ
                                ] * beta[t+1][zJ] * emisja[zJ][ obs[t+1] ]
                        for zI in range(N) ] ) for zJ in range(N) ] )
            if mianownik == 0:
                print('mianownik XI (dla t='+str(t)+') = ', mianownik )
            for zt in range(N):
                for ztplus1 in range(N):
                    if mianownik > 0:
                        xi[t][(zt,ztplus1)] = alfa[t][zt] * przejscia[zt][ztplus1] * beta[t+1][ztplus1
                        ] * emisja[ztplus1][ obs[t+1]
                        ] / mianownik
                    else:
                        xi[t][(zt,ztplus1)] = 0

        pi2 = [ eta[0][i] for i in range(N) ]    # teoretycznie miało by być tak, ale następny wiersz czasem działa lepiej
        #pi2 = [ (eta[0][i]+eta[1][i])/2 for i in range(N) ]
           
        przejscia2 = np.array( [[ sum( xi[t][(i,j)] for t in range(T-1)
                                      ) / sum( eta[t][i] for t in range(T-1) )
                      for j in range(N) ] for i in range(N) ] )
        
        emisja2 = np.array( [[ sum( unit( o, obs[t] ) * eta[t][i] for t in range(T)
                                   ) / sum( eta[t][i] for t in range(T) )
                      for o in range(NO) ] for i in range(N) ] )


        
        if steps > 1:
            alfa2, alfaAll2 = forward( obs, pi2, przejscia2, emisja2 )
            alfa1, alfaAll1 = forward( obs, pi, przejscia, emisja )            
            diff = abs( alfaAll2 - alfaAll1)/alfaAll2
            if shouldPrint:                
                print('DIFF = ',diff,'\t\t prawd.obs. = ', alfaAll2) 
            if diff < 1e-10:
                break        

        pi, przejscia, emisja = pi2, przejscia2, emisja2
        step += 1
        if step == 1 and steps > 1:
            print('Maks.',steps,'iteracji: ',end='\t')
        if step % 100 == 0 and step > 0:
            print( step, end='\t' )
    if step > 1:
        print('BaumWelchOneObs: \n')
        print( 'iteracji =',step)
        printParams( pi, przejscia, emisja)
        print('prawdop. obserwacji względem znaleźionych parametrów modelu = ', alfaAll2 )
    return pi, przejscia, emisja, eta, xi

def BaumWelchManyObs(obsLista, priorPi, przejsciaM, emisjaE, steps = 10, shouldPrint=False):
    def unit( i, j ):
        if i == j:
            return 1
        else:
            return 0

    pi = deepcopy(priorPi)
    przejscia = deepcopy(przejsciaM)
    emisja = deepcopy(emisjaE)
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

        #pi2 = [ sum( etaRazem[r][0][i] for r in range(R) ) for i in range(N) ]     # teoretycznie miało by być tak, ale następny wiersz działa lepiej
        pi2 = [ sum( (etaRazem[r][0][i]+etaRazem[r][1][i])/2 for r in range(R) ) for i in range(N) ]
        #normalizacja pi:
        pi2 = [ p/np.sum(pi2) for p in pi2 ]
        przejscia2 = np.array( [[
            sum( sum( xiRazem[r][t][(i,j)] for t in range(T-1) ) for r in range(R)
                 )/ sum( sum( etaRazem[r][t][i] for t in range(T-1) ) for r in range(R)
                         ) for j in range(N) ] for i in range(N) ] )
        emisja2 = np.array( [[
            sum( sum( unit( o, obsLista[r][t] ) * etaRazem[r][t][i] for t in range(T) ) for r in range(R)
                 ) / sum( sum( etaRazem[r][t][i] for t in range(T) ) for r in range(R)
                          ) for o in range(NO) ] for i in range(N) ] )



        diff, suma = 0, 0
        for obs in obsLista:
            alfa2, alfaAll2 = forward( obs, pi2, przejscia2, emisja2 )
            alfa1, alfaAll1 = forward( obs, pi, przejscia, emisja )
            suma += abs( alfaAll2 )
            if abs( alfaAll2 - alfaAll1) > diff:
                diff = abs( alfaAll2 - alfaAll1)
                
        if shouldPrint:
            print('DIFF = ',alfaAll2,'-',alfaAll1,' =', diff*len(obsLista)/suma )
        if diff*len(obsLista)/suma < 0.0000001:   # kiedy względne ulepszenie mniejsze od 1e-7
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
            print('Maks.',steps,'iteracji: ',end='\t')
        if step % 100 == 0 and step > 0:
            print( step, end='\t' )
    print('\nBaumWelchManyObs: \n')
    print( 'iteracji =',step)
    printParams( pi, przejscia, emisja)
    print('średnia prawdop. obserwacji z sekwencji obserwacyjnej względem znaleźionych parametrów modelu = ', suma/len(obsLista))
    return pi, przejscia, emisja, suma/len(obsLista)



if __name__ == '__main__':
    
    #obs = [0,1,2]
    obs =  [1,1,1,2,2,2,1,1,0,1,2,2 ]

    #start_probability = {'Healthy': 0.6, 'Fever': 0.4}
    prior = [0.34, 0.33, 0.33]
     
    #transition_probability = {
    #   'Healthy' : {'Healthy': 0.69, 'Fever': 0.3, 'E': 0.01},
    #   'Fever' : {'Healthy': 0.4, 'Fever': 0.59, 'E': 0.01},
    #   }
    przejscia = np.array( [
        [ 0.6, 0.2,  0.2 ],
        [ 0.1,  0.8, 0.1 ],
        [ 0.1, 0.1, 0.8 ]
        ] )
     
    #emission_probability = {
    #   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
    #   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6},
    #   }
    emisja = np.array( [
        [ 0.8,  0.1,  0.1 ],
        [ 0.1,  0.8,  0.1 ],
        [ 0.1,  0.1,  0.8 ]
        ] )

    # ukryte =  [1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 2, 2, 2, 1]
    #obs = [1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 2, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 1, 1, 2, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 2, 0, 2, 0, 2, 2, 2, 0]
    #obs = [2, 0, 1, 1, 0, 0, 2, 2, 2, 2, 1, 0, 0, 0, 0, 2, 2, 2, 1, 2, 2]
    #obs = [0, 0, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]
    prior0 = [0.5, 0.0, 0.5]
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
    
    print('obs =\t',obs)
    pi, prz, emi, eta, xi = BaumWelchOneObs(obs, prior0, przejscia0, emisja0, steps = 20000, shouldPrint=False)
    printParams( pi, prz, emi)

    #obsLista = [[1, 1, 1, 2, 1, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 2, 1, 2, 0, 1, 2, 2], [0, 2, 0, 2, 1, 2, 2, 2, 2, 2, 0], [0, 2, 2, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1], [0, 0, 0, 0, 2, 1, 2, 2, 2, 1, 2, 2, 1, 2, 2, 0, 2, 0, 0, 2, 1, 2, 2, 2, 0, 2, 2, 1], [0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 0, 0, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0], [0, 2, 0, 0, 2, 2, 2, 2, 0, 0, 1, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2], [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 1], [0, 0, 0, 0, 1, 2, 1, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 2, 2, 2, 0, 2, 2, 1, 1, 1, 1], [0, 2, 0, 1, 0, 2, 2, 1, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 2], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 2, 2, 2, 1], [0, 2, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2], [0, 0, 0, 0, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 0, 2], [2, 0, 0, 0, 0, 1, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 2, 0, 1], [1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 0, 0, 0], [2, 0, 0, 1, 2, 1, 0, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 2], [1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 1, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0]]
    obsLista = [[0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1], [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 2, 2, 0, 2, 2, 2, 2, 0, 0], [0, 0, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 0, 0, 0, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 0], [1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 1, 2, 2], [1, 1, 1, 1, 1, 1, 0, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1], [1, 1, 2, 2, 2, 2, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 0, 0], [0, 2, 1, 1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0], [0, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 0, 0, 2, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1], [1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0], [0, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 1, 1], [1, 1, 0, 2, 2, 2, 2, 1, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 0], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 2, 1, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1], [0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 1, 2, 2, 2, 1], [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2], [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 2, 2, 1, 1, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 0, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1, 0, 2, 2, 0, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1], [1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2], [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 0, 1, 0, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 2, 2, 0, 1, 2, 2, 0, 0, 2, 2], [0, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 2, 2, 2, 1, 1, 1, 2], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1], [1, 1, 1, 1, 2, 2, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0], [1, 2, 2, 0, 0, 0, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 1, 0, 2, 2, 0, 0, 1, 0, 0, 1, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2], [0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2], [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 0, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 0, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0, 0], [0, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 2], [0, 0, 2, 2, 2, 2, 2, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1], [1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 2, 2, 0, 0, 2, 2, 2, 2, 0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 1, 1, 1, 1, 2], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2], [0, 0, 2, 0, 2, 2, 0, 0, 0, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
    print('\nobserwacyjna sekwencja: ')
    for o in obsLista[:5]:
        print(o)
    print()
    BaumWelchManyObs(obsLista[:5], prior0, przejscia0, emisja0, steps = 2000, shouldPrint=False)

