import numpy as np

# metoda 'do tyłu', inaczej 'cofająca się' (angl. 'backward')
def backward(obs, przejsciaM, emisjaE, priorPi ):
    # obs - sekwencja obserwacji o długości 'n'
    # przejscia - macierz prawdopodobieństw przejść (od x[i] do x[j])
    # emisja - macierz przejść emisji (od z[i] do x[j])
    # priorPi - prawdodpodbieństwa początkowe ( pi[i] )
    # zwraca słownik z prawdopodobieństwem warunkowym postaci:
    # { ..., z[i]:  P( o[1:n] | z[i] ), ...}

    N = len(przejsciaM)    # ilośc stanow procesu Z = ilości wierszy macierzy przejść
    NO = len(emisjaE[0])     #ilość stanów obserwowanych = ilości kolumn macierzy emisji
    T = len(obs)       # obs[t-1] -> O[t] we wzorze
    O = [''] + obs
    beta = [{} for K in range(T+1)]
    
    for zJ in range(N):
        beta[T][zJ] = 1.0   # dla niektórych zagadnień przyrównuje się do =emisjaE[zJ][ O[T] ]

    for t in reversed(range(1,T)):
        for zt in range(N):
            beta[t][zt] = sum((beta[t+1][ztplus1]
                              * przejsciaM[zt][ztplus1] * emisjaE[ztplus1][O[t+1]] )
                             for ztplus1 in range(N))
    for zt in range(N):
        beta[0][zt] = priorPi[zt] * emisjaE[zt][ O[1] ] * beta[1][zt]
        '''
        beta[0][zt] = sum((beta[0+1][ztplus1]
                              * priorPi[ztplus1] * emisjaE[ztplus1][O[0+1]] )
                             for ztplus1 in range(N))
        '''

    return beta, sum( beta[0][zt] for zt in range(N) )

####### P( Z[k] | O ) jest proporcjonalne prawd. złożonemu P( Z[k], O[1:k], O[k+1:n] ), a znaczy
####### P( Z[k] | O ) jest proporcjonalne iloczynu P( O[k+1:n] | Z[k], O[1:k] ) * P( Z[k], O[1:k] ), skąd
####### wiedząc o niezależności O[k+1:n] od O[1:k]|Z[k] i  korzystając z własności procesów Markowa mamy, że:
####### P( Z[k] | O ) jest proporcjonalne iloczynu P( O[k+1:n] | Z[k] ) * P( Z[k], O[1:k] )



if __name__ == "__main__":
    #'''    
    obs = [0, 1,2,1]
    prior = np.array( [0.5, 0.5 ] )
    przejscia = np.array( [
        [ 0.3, 0.7 ],
        [ 0.5, 0.5 ]
        ] )
    emisja = np.array( [
        [ 0.5,  0.5,  0.0 ],
        [ 0.5,  0.0,  0.5 ]
        ] )
    '''
    obs = [1,2,1]
    prior = np.array( [0.5, 0.5 ] )
    przejscia = np.array( [
        [ 1.0, 0.0 ],
        [ 0.0, 1.0 ]
        ] )
    emisja = np.array( [
        [ 1.0,  0.0,  0.0 ],
        [ 0.0,  0.5,  0.5 ]
        ] )
    '''
    print('\n######## backward ##########\n')
    backw, prObsBeta = backward( obs, przejscia, emisja, prior )
    print('początkowe prawdopodopbieństwa:', prior )
    print('pr.przejścia:')
    for p in przejscia:
        print(p)
    print()
    print('pr.emisji:')
    for p in emisja:
        print(p)
    print()

    print('obserwacje: ',obs)
    print()
    print('absolutne prawdopodobieństwa warunkowe postaci \n{ ..., z[i]:  P( o[t+1:T] | z[i]) ), ...} :\n')
    i = 0
    for elem in backw:
        if i > 0:
            print('krok',i,' ',elem)
        i += 1

    print('\n[ beta[1]...beta[T] ] =', backw[1:])
    print('beta[0] = ', backw[0] )
    print('prawdopodobieństwo obserwacyjnej sekwencji z "do tyłu" = ',
          prObsBeta)
