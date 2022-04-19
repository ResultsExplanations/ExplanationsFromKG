import utils
import Algorithms




def main():

    SOEX1()

def SOEX1():
    X, Y, E = utils.SOEX1(1, 467)
    E = E.astype(int)

    ans = Algorithms.MCIMR(E, X, Y, 5, True)
    print("explanation: ", ans)
    Eans = E[list(ans)]
    for e in ans:
        print("responsibility: ",e, Algorithms.responsibility(X, Y, Eans, e))


if __name__ == '__main__':
    main()