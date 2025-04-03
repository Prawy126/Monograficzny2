import numpy as np

from _4a_doPrzodu_forward import forward
from _4b_doTylu_backward import backward


if __name__ == "__main__":

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
    obs = [1,2,1,2]
    prior = np.array( [0.5, 0.5 ] )
    przejscia = np.array( [
        [ 0.8, 0.2 ],
        [ 0.2, 0.8 ]
        ] )
    emisja = np.array( [
        [ 1.0,  0.0,  0.0 ],
        [ 0.0,  0.5,  0.5 ]
        ] )
    '''

    obs = [2]*12
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
    alfa, prObsAlfa = forward( obs, przejscia, emisja, prior  )
    beta, prObsBeta = backward( obs, przejscia, emisja, prior )
    print('prawdopodobieństwo obserwacyjnej sekwencji z "do przodu" = ',
          round(prObsAlfa, 10))
    print('prawdopodobieństwo obserwacyjnej sekwencji z "do tyłu" = ',
          round(prObsBeta, 10))
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
    print('### forward ###')
    print('prawdopodobieństwa złożone "alfa" postaci \n{ ..., {z[i]:  P( z[i], o[1:t] ), ...} :\n' \
          +'inaczej postaci: { ..., { 0:  P(0, o[1:t]), 1:  P(1, o[1:t])... N-1:  P(N-1, o[1:t]), ...} :\n')
    for i,elem in enumerate(alfa):
        #if i > 0:
        print('krok',i,' ',elem)

    print(' P( Z[1:n] | O[1:n] ) = ', prObsAlfa )

    print('### backward ###')
    print('warunkowe prawdopodobieństwa "beta" postaci \n{ ..., z[i]:  P( o[t+1:T] | z[i]), ...} :\n')
    for i,elem in enumerate(beta):
        #if i > 0:
        print('krok',i,' ', elem )
            
    print()
    print('### forward-backward ###')
    print()

    # gamma = doprecyzowane warunkowe prawdopodobieństwo dla każdego stanu na każdym kroku 't' względem obserwacji
    print('warunkowe prawdopodobieństwa "gamma" postaci \n{ ..., z[i]:  P( z[i] | o[1:t] ), ...} :\n')    
    gamma = [{} for i in range(len(obs)+1) ]
    for m in range(1,len(obs)+1):
        for zJ in range(len(przejscia)):
            if prObsAlfa == 0:
                gamma[m][zJ] = 0.0
            else:
                gamma[m][zJ] = alfa[m][zJ] * beta[m][zJ] / prObsAlfa
    # zerowy krok w 'doPrzodu' i 'doTyłu'
    # obliczamy za pomocą priorów, a nie macierzy przejścia
    # dlatego zerowy wiersz w 'gamma' nie normalizuje się automatycznie
    # przy dzieleniu na P( O[0:T] ) == prObs ( <== zgrubsza ze wzoru P(A,B) = P(A|B)*P(B) )

    for i,elem in enumerate(gamma):
        if i > 0:
            print('krok',i,' ', elem)
    print()
                                                           
