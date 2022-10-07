import utils
import Algorithms
import time




def main():
    #an example of generating explanation to SO query #1
    SOEX1()

def SOEX1():
    X, Y, E = utils.SOEX1(1, 467)
    E = E.astype(int)
    EE = utils.preprosees(E)

    #only offline pruning
    ans = Algorithms.MCIMR(EE, X, Y, 5, True)
    print("explanation: ", ans)
    Eans = E[list(ans)]
    for e in ans:
        print("responsibility: ",e, Algorithms.responsibility(X, Y, Eans, e))

    #online pruning
    start = time.time()
    EE = utils.onlineFiltering(X,Y,EE)
    end = time.time()
    ans = Algorithms.MCIMR(EE, X, Y, 5, True)
    print("explanation: ", ans, end-start)
    Eans = E[list(ans)]
    for e in ans:
        print("responsibility: ", e, Algorithms.responsibility(X, Y, Eans, e))


if __name__ == '__main__':
    main()
