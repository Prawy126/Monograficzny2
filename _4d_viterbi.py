import numpy as np

##### Algorytm zwraca:
##### trajektoria - najbardziej prawdopodobna ścieżka ukrytych stanów ze względu na obserwacje;
##### T1 -  prawdopodobieństwo najbardziej prawdopodobnej ścieżki ukrytych stanów na odpowiednią chwile;
##### T2 - x[j-1] najbardziej prawdopodobną ścieżke na odpowiednią chwile;
#####
##### Suma (wzg. X) od P(X,Z)P(Y) == P(Y) 
#####
##### Ponieważ P(Z|X) jest proporcjonalna P(X,Z)==P(Z,X) (bo P(Z|X)= P(Z,X)/P(X) ),
##### a my szukamy arg-maksimum, to
##### mi[k] = argmax( po Z[1:k-1] ) od P( Z[1:k],X[1:k] ) jest proporcjonalne
##### do P( X[1:k] | Z[1:k] ), co (bez argumentacji) jest proporcjonalne
##### do argmax( po Z[1:k-1] ) od P(X[k] | Z[k])*P( Z[k] | Z[k-1] ), skąd mamy
##### mi[k] = argmax( po Z[1:k-1] ) P(X[k] | Z[k]) P( Z[k] | Z[k-1] ) P( Z[1:k-1],X[1:k-1] )
##### mi[k] = argmax( po Z[1:k-1] ) P(X[k] | Z[k]) P( Z[k] | Z[k-1] ) * mi[k-1]


def viterbi(obs, przejscia, emisja, priorPi=None):
    # obs - sekwencja obserwacji o długości 'n'
    # przejscia - macierz prawdopodobieństw przejść (od x[i] do x[j])
    # emisja - macierz przejść emisji (od z[i] do x[j])
    # priorPi - prawdodpodbieństwa początkowe ( pi[i] )
    # zwraca 3 argumenty:
    # 1) 'x' - najbardziej prawdopodobną estymacje trajektorii ukrytych stanów,
    # uwarunkowanych danymi obserwacji (i pod warunkami przejść, emisji i pirorów)
    # 2) prawdopodobieństwo najbardziej prawdopodobnej trajektorii (na każdym kroku)
    # 3) 'x_j-1' naj.prawd. trajektorii

    # wymiar przestrzeni stanów
    N = len(przejscia)

    #emisja = np.repeat(emisja[np.newaxis, :], K, axis=0)

    # inicjalizacja prawdopodobieństw początkowych na kroku 0, jak jednakowe, jeśli None
    priorPi = priorPi if priorPi is not None else np.full(N, 1 / N)
    T = len(obs)
    T1 = np.empty((N, T), 'd')  # 'd' - double-precision floating-point number      - zmienic do stanyM
    T2 = np.empty((N, T), 'B')  # 'B' - unsigned byte                                       -zmienic do obserwacje naj.prawd. trajektorii

    # inicjalizacja tablic z pierwszej obserwacji
    T1[:, 0] = priorPi * emisja[:, obs[0]]
    T2[:, 0] = 0

    # iterujemy po obserwacjach, aktualizując tablicy
    for i in range(1, T):
        T1[:, i] = np.max( T1[:, i - 1] * przejscia.T * emisja[np.newaxis, :, obs[i]].T, 1)
        T2[:, i] = np.argmax(T1[:, i - 1] * przejscia.T, 1)
        # 'argmax' zwraca numer wiersza z maksimum w kolumnie '1'
        print( np.round(T1,4))
        print()

    # konstruujemy optymalną model trajektorii
    trajektoria = np.empty(T, 'B')    # 'B' = unsigned byte
    trajektoria[-1] = np.argmax(T1[:, T - 1])     # zwraca numer wiersza z maksimum w kolumnie 'T-1'
    for i in reversed(range(1, T)):
        trajektoria[i - 1] = T2[trajektoria[i], i]

    return trajektoria, T1, T2

def normalizedProbList( probs ):
    if sum([p for p in probs ]) != 0:
        return [ p/sum([p for p in probs ]) for p in probs ]
    else:
        return probs


################


prior = [0.4, 0.2, 0.4]
przejscia = np.array( [
    [ 0.5, 0.3,  0.2],
    [ 0.4,  0.5, 0.1 ],
    [ 0.1,  0.8, 0.1 ]
    ] )
emisja = np.array( [
    [ 0.5,  0.4,  0.1 ],
    [ 0.1,  0.5,  0.4 ],
    [ 0.2,  0.2,  0.6 ]
    ] )

#obs = [0,1,2]
obs = [2,1,2,2,2]

#obs = [1,1,1,2,2,2,1,1,0,1,2,2 ]  #-> [1 1 1 2 2 2 1 1 1 1 2 2]
#obs = [1,1,1,0,0,0,1,1,2,1,0,0 ]  #->  [1 1 1 1 1 1 1 1 1 1 1 1]
#obs = [1,1,2,2,1,1,1,1,0,1,1,0 ]  #->  [1 1 2 2 1 1 1 1 1 1 1 1]
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

trajektoria, T1, T2 = viterbi(obs, przejscia, emisja, prior)

print('najbardziej prawdopodobna ścieżka ukrytych stanów ze względu na obserwacje: ')
print( trajektoria)
print('\nprawdopodobieństwa najbardziej prawdopodobnej ścieżki ukrytych stanów \n na odpowiednią chwile, macierz T1 =')
print( np.round(T1,2))
print('\ntrajektoria[j-1] najbardziej prawdopodobną ścieżke \n na odpowiednią chwile, macierz T2 =')
print( np.round(T2,2))

print()
print('T1 z normalizowanymi prawdopodobieństwami:')
for x in T1:
    print( [ round(x,2) for x in normalizedProbList(x) ] )
print('T2:')
for x in T2:
    print( x )

print('\n\n##### drugi przykład ########')
prior = np.array([0.5, 0.5])
przejscia = np.array([
    [0.75, 0.25],
    [0.32, 0.68]])
emisja = np.array([
    [0.8, 0.1, 0.1],
    [0.1, 0.2, 0.7]])
obs = [0, 1, 2, 1, 0]

x, T1, T2 = viterbi(obs, przejscia, emisja) #, prior)

print('najbardziej prawdopodobna ścieżka ukrytych stanów ze względu na obserwacje: ')
print( x)
print('\nprawdopodobieństwa najbardziej prawdopodobnej ścieżki ukrytych stanów \n na odpowiednią chwile, macierz T1 =')
print(  np.round(T1,2))
print('\ntrajektoria[j-1] najbardziej prawdopodobną ścieżke \n na odpowiednią chwile, macierz T2 =')
print(  np.round(T2,2))

