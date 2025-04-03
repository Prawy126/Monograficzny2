import numpy as np

# metoda 'do przodu', inaczej 'postępowa' (ang. 'forward')
def forward(obs, przejsciaM, emisjaE, priorPi):
    # obs - sekwencja obserwacji o długości 'n'
    # przejscia - macierz prawdopodobieństw przejść (od z[i] do z[j])
    # emisja - macierz przejść emisji (od z[i] do o[j])
    # priorPi - prawdodpodbieństwa początkowe ( pi[i] )
    # zwraca słownik z prawdopodobieństwem łącznym postaci:
    # { ..., z[i]:  P( z[i], o[0:T-1] ), ...}
    # inaczej { ..., z[i]:  alfa[ z[i] ][ o[0:T-1] ), ...}
    N = len(przejsciaM)    # ilośc stanow procesu Z = ilości wierszy macierzy przejść
    NO = len(emisjaE[0])     #ilość stanów obserwowanych = ilości kolumn macierzy emisji
    T = len(obs)       
    O = [''] + obs       # wtedy obs[t-1] -> O[t] we wzorze,  O[0]=='' nigdzie nie używa się
    alfa = [{},{}]      # alfa[0] - nie używa się
    
    for zI in range(N):
        alfa[1][zI] = priorPi[zI] * emisjaE[zI][ O[1] ]

    for t in range(2, T+1):
        alfa.append({})
        for zt in range(N):
            alfa[t][zt] = emisjaE[zt][O[t]] * sum( [   # O[t]==obs[t-1]
                przejsciaM[ztminus1][zt] * alfa[t-1][ztminus1]
                for ztminus1 in range(N) ])
            # suma po wszystkich wartościach 'ztminus1', czyli od 0 do N-1
    prawdopobienstwoObserwacji = sum( alfa[-1][stan] for stan in range(N) )

    return alfa, prawdopobienstwoObserwacji

def normalizedProb( alfa ):
    if sum([alfa[k] for k in alfa.keys()]) == 0:
        return alfa, 0      # wtedy wszystkie alfa[k] są równe zeru - sekwencja obs. niemożliwa
    else:
        alfa_denom = sum([alfa[k] for k in alfa.keys()])
        return { k:(alfa[k]/alfa_denom) for k in alfa.keys() }, alfa_denom



if __name__ == "__main__":
        
    obs = [1, 2, 1]
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
    obs = [1]*20
    prior = np.array([ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
    przejscia = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ])
    emisja = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0],
        ])
    '''

    obs = [2 ]*6 +[1]*6
    prior = [ 1/3, 1/3, 1/3 ]
    # oceny: dobre - 0, średnie - 1, słabe - 2
    przejscia  = np.array([
        [ 0.5, 0.3, 0.2 ],
        [ 0.2, 0.6, 0.2],
        [ 0.1, 0.4, 0.5],
        ])
    # wytraty: wysokie - 0, średnie - 1, niskie - 2
    emisja = np.array([
        [ 0.25, 0.5, 0.25 ],
        [ 0.35, 0.5, 0.15],
        [ 0.4, 0.2, 0.4],
        ])
    
    print('\n######## forward ##########\n')
    forw, forwAll = forward( obs, przejscia, emisja, prior )
    print('początkowe prawdopodobieństwa:', prior )
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
    print('absolutne prawdopodobieństwa złożone postaci \n{ ..., z[i]:  P( z[i], o[1:n] ), ...} :\n')
    i = 0
    for elem in forw:
        if  i > 0:
            print('krok',i,' ',elem)
        i += 1

    print('\n[ alfa[1]...alfa[T] ] =', forw[1:])
    print('alfa[0] = ', forw[0] )

    print('prawdopodobieństwo obserwacji ', obs,' dla danego modelu wynosi =', forwAll)
    print()
    print('normalizowane prawdopodobieństwa:')
    for i,elem in enumerate(forw):
        if i > 0:
            print('krok',i,' ', normalizedProb(elem)[0] )
    print()

