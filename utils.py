import pandas as pd
PATH = "C:/Users/brity/OneDrive/Desktop/visualization project/"
from pyitlib import discrete_random_variable as drv
import varclushi
from scipy.stats import spearmanr
import info_theo as info
import random



def clusters(E):
    names = []
    clusters = varclushi.VarClusHi(E, maxeigval2=.7, maxclus=None)
    clusters.varclus()
    #print(clusters.rsquare)
    clustersDF =  clusters.rsquare.groupby(["Cluster"])
    for n,g in clustersDF:
        x = g.loc[g['RS_Ratio'].idxmax()]["Variable"]
        #print(n,x, len(g), g["Variable"].tolist())
        names.append(x)

    EE = E[names]
    print(len(E.columns), len(EE.columns))
    return EE






def countIndependent(Y,E):
    Y = [int(ee) for ee in Y]
    EE = []
    # first iteration
    for e in E:
        varE = E[e].tolist()
        varE = [int(ee) for ee in varE]
        count = varE.count(-1)
        if count > 0:
            indices = [i for i, x in enumerate(varE) if x == -1]
            varEE = [x for i, x in enumerate(varE) if not i in indices]
            YY = [x for i, x in enumerate(Y) if not i in indices]
            if len(YY) == 0:
                EE.append(e)
            try:
                valE = drv.information_mutual(YY, varEE)
            except:
                EE.append(e)
            #print(valE,e)
        else:
            valE = drv.information_mutual(Y, varE)

        if valE < 0.01:
            EE.append(e)
            #print(valE, e)
    return EE

def countXORAtts(E,EE,Y):
    Y = [int(ee) for ee in Y]
    EEE = []
    # first iteration
    for e_ in EE:
        for e in E:
            varE = E[e].tolist()
            varE = [int(ee) for ee in varE]
            count = varE.count(-1)
            if count > 0:
                indices = [i for i, x in enumerate(varE) if x == -1]
                varEE = [x for i, x in enumerate(varE) if not i in indices]
                YY = [x for i, x in enumerate(Y) if not i in indices]
                ee_ = [x for i, x in enumerate(e_) if not i in indices]
                if len(YY) == 0:
                    continue
            count = ee_.count(-1)
            if count > 0:
                indices = [i for i, x in enumerate(ee_) if x == -1]
                varEE = [x for i, x in enumerate(varEE) if not i in indices]
                YY = [x for i, x in enumerate(YY) if not i in indices]
                ee_ = [x for i, x in enumerate(ee_) if not i in indices]
                if len(YY) == 0:
                    continue
                if len(ee_) == 0:
                    continue
            if len(YY) == 0:
                continue
            if len(ee_) == 0:
                continue
            valE = drv.information_mutual_conditional(YY,ee_, varEE)
            #print(valE)
            if valE > 0.01:
                if not e_ in EEE:
                    EEE.append(e_)
    return EEE

def CheckXOR(Y,E):
    EE = countIndependent(Y,E)
    print("num of attributes: ", len(E.columns))
    print("num of attributes independent of Y: ", len(EE))
    EEE = countXORAtts(E,EE,Y)
    print("num of 2-way XOR: ", len(EEE))



def filterColumns(E):
    numRows = len(E)
    to_drop = []
    # filter by high entropy
    black = get_high_entropy_atts(E)
    for e in black:
        if not e in to_drop:
            to_drop.append(e)

    print("high entropy: ",len(black))

    countS = 0
    for col in E:
        if col in to_drop:
            continue
        if "Unnamed" in col:
            to_drop.append(col)
            countS = countS +1
            continue
        if len(set(E[col].tolist())) < 2:
            to_drop.append(col)
            countS = countS + 1
            continue
        E[col] = E[col].fillna(-1)
        varE = E[col].tolist()
        varE = [int(ee) for ee in varE]
        count = varE.count(-1)
        if count >= 0.9 * numRows:
            to_drop.append(col)
            countS = countS + 1
            continue
    print("simple filtering: ",countS)



    return to_drop

def get_high_entropy_atts(data, start=0.3, end=0.4, steps=0.01, cut=0.0001, alpha=0.01, debug=False):
    # treatment="'"+treatment+"'"
    samplesizez = []
    dic = dict()
    black = []
    white = []
    basesample = data.sample(frac=0.3, replace=False)
    features=data.columns.values
    size = len(data.index)
    while start <= end:
        start = start + steps
        sample = basesample.sample(frac=start, replace=False)
        samplesizez.insert(0, (len(sample.index)))
        inf=info.Info(sample)
        for col in features:
            if col in dic:
                list = dic[col]
                list.insert(0, inf.entropy(col, size))
                dic[col] = list
            else:
                dic[col] = [inf.entropy(col, size)]

    for col in features:
        if not any(dic[col]):
            continue
        rho, pval1 = spearmanr(samplesizez, dic[col])

        if pval1 <= alpha:
            black.insert(0, col)
        else:
            white.insert(0, col)
    # self.features=np.array(white)
    return black


def preprosees(E):
    print("num of columns: ", len(E.columns))
    to_drop = filterColumns(E)

    EE = E.drop(to_drop, axis=1)
    print("num of columns after offline filtering: ", len(EE.columns))

    return EE

def onlineFiltering(X,Y,E, verbose= False):
    if verbose:
        print("num of columns: ", len(E.columns))

    to_drop = filterColumnsOnline(E, X, Y)
    EE = E.drop(to_drop, axis=1)
    if verbose:
        print("num of columns after online filtering: ", len(EE.columns))

    # EEE = clusters(EE)
    # print("num of columns after clustering: ", len(EEE.columns))
    return EE

def filterColumnsOnline(E, X, Y):
    # inf = info.Info(E)

    Y = [int(i) for i in Y]
    X = [int(i) for i in X]
    to_drop = []
    countS = 0
    for col in E:
        varE = E[col].tolist()
        varE = [int(ee) for ee in varE]
        indices = [i for i, x in enumerate(varE) if x == -1]
        varEE = [x for i, x in enumerate(varE) if not i in indices]
        YY = [x for i, x in enumerate(Y) if not i in indices]
        XX = [x for i, x in enumerate(X) if not i in indices]
        if len(YY) == 0 or len(XX) == 0:
            continue


        chXE = drv.entropy_conditional(XX,varEE)#inf.CH(XX,varEE)#MCMR.cEntropy(XX, varEE)
        if chXE > 0.01:
            continue
        chEX = drv.entropy_conditional(varEE,XX)#inf.CH(varEE,XX)#MCMR.cEntropy(varEE, XX)
        if chEX <= 0.01:
            countS = countS + 1
            to_drop.append(col)
            continue
    #print("FD: ", countS)

    # filter by relevance
    EE = E.drop(to_drop, axis=1)
    black = countIndependent(Y, EE)
    for e in black:
        if not e in to_drop:
            to_drop.append(e)

    #print("low relevance: ", len(black))
    return to_drop


def covideEX3(rows, ncolumns):
    df = pd.read_csv(PATH+"/data/CovidEX3Prepro.csv")

    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)

    Y = result["Deaths / 100 Cases"]
    X = result["WHO Region"]
    del result["Deaths / 100 Cases"]
    del result["WHO Region"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E


def covideEX2(rows, ncolumns):
    df = pd.read_csv(PATH+"/data/CovidEX2Prepro.csv")

    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)

    Y = result["Deaths / 100 Cases"]
    X = result["Country"]
    del result["Deaths / 100 Cases"]
    del result["Country"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E

def covideEX1(rows, ncolumns):
    df = pd.read_csv(PATH+"/data/CovidEX1Prepro.csv")
    #df = pd.read_csv(PATH + "/country_wise_latest.csv")

    result = df.sample(frac=rows, replace=True, random_state=1)
    #

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)
        dataTypeObj = result.dtypes[col]
        if dataTypeObj == "object":
            result[col] = result[col].astype('category')
            result[col] = result[col].cat.codes

    Y = result["Deaths / 100 Cases"]
    X = result["Country"]
    del result["Deaths / 100 Cases"]
    del result["Country"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E

def forbesDirectors(rows,ncolumns):


    df = pd.read_csv(PATH + "/data/examples/forbesDirectorsPrepro.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)

    Y = result["Pay (USD millions)"]
    X = result['Name']
    del result["Pay (USD millions)"]
    del result["Name"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E

def forbesAthlese(rows,ncolumns):


    df = pd.read_csv(PATH + "/data/examples/forbesAthletesPrepro.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)

    Y = result["Pay (USD millions)"]
    X = result['Name']
    del result["Pay (USD millions)"]
    del result["Name"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E

def forbesActors(rows, ncolumns):

    df = pd.read_csv(PATH + "/data/examples/forbesActorsPrepro.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)

    Y = result["Pay (USD millions)"]
    X = result['Name']
    del result["Pay (USD millions)"]
    del result["Name"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E


def flightsCA(rows, ncolumns):

    df = pd.read_csv(PATH + "/data/flightsCAPrepro.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)

    Y = result["AVG_DELAY"]
    X = result['City']
    del result["AVG_DELAY"]
    del result["City"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E


def flightsEX4(rows,ncolumns):

    df = pd.read_csv(PATH+"/data/FlightsEX4Prepro.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)

    Y = result["AVG_DELAY"]
    X = result[['AIRLINE','STATE']]
    del result["AVG_DELAY"]
    del result["AIRLINE"]
    del result["STATE"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E


def flightsEX3(rows,ncolumns):

    df = pd.read_csv(PATH+"/data/FlightsEX3Prepro.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)

    Y = result["AVG_DELAY"]
    X = result['AIRLINE']
    del result["AVG_DELAY"]
    del result["AIRLINE"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E

def flightsEX2(rows,ncolumns):

    df = pd.read_csv(PATH + "/data/examples/FlightsEX2Prepro.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)

    Y = result["AVG_DELAY"]
    X = result['STATE']
    del result["AVG_DELAY"]
    del result["STATE"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E

def flightsEX1(rows, ncolumns):
    df = pd.read_csv(PATH + "/data/examples/flightsEx1CitiesPrepro.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)
        dataTypeObj = result.dtypes[col]
        if dataTypeObj == "object":
            result[col] = result[col].astype('category')
            result[col] = result[col].cat.codes

    Y = result["DEPARTURE_DELAY"]#result["AVG_DELAY"]
    X = result["ORIGIN_AIRPORT"]#result['AIRPORT']
    del result["DEPARTURE_DELAY"]#result["AVG_DELAY"]
    del result["ORIGIN_AIRPORT"]#result["AIRPORT"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]
    return X, Y, E



def SOEX12HOPS(rows, ncolumns):

    df = pd.read_csv(PATH+"/data/SOEX12HOPSPrepro.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:

        result[col] = result[col].fillna(-1)
    X = result["Country"]
    Y = result['Salary']
    del result["Country"]
    del result["Salary"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]

    return X, Y, E

def SORandom():
    df = pd.read_csv(PATH + "/data/examples/SOEx1PreproFull.csv")

    df = df.sample(frac=0.3, replace=True, random_state=1)
    df = df.infer_objects()
    for col in df:
        df[col] = df[col].fillna(-1)

    Exp = ["Country", "Continent"]
    WHERE = {"Hobby":[1, 0], "Student":[0, 1,2]}
    e = random.choice(Exp)
    where_a = random.choice(list(WHERE.keys()))
    where_v = random.choice(WHERE[where_a])
    print("exposure: ", e,"where: ", where_a, where_v)

    result = df[df[where_a].isin([where_v])]
    print(len(df), len(result))


    X = result[e]
    Y = result['ConvertedSalary']
    del result[e]
    del result["ConvertedSalary"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]

    return X, Y, E

def SOEX1(rows, ncolumns):

    df = pd.read_csv(PATH + "/data/examples/SOEx1Prepro.csv")
    #df = pd.read_csv(PATH + "/data/examples/SOEx1PreproFull.csv")
    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:

        result[col] = result[col].fillna(-1)
    X = result["Country"]
    Y = result['ConvertedSalary']
    del result["Country"]
    del result["ConvertedSalary"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]

    return X, Y, E

def SOEX2(rows, ncolumns):
    df = pd.read_csv(PATH + "/data/examples/SOEx2Prepro.csv")

    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)
    X = result["Continent"]
    Y = result['Salary']
    del result["Continent"]
    del result["Salary"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]

    return X, Y, E

def SOEX3(rows, ncolumns):
    df = pd.read_csv(PATH+"/data/examples/SOEx3Prepro.csv")

    result = df.sample(frac=rows, replace=True, random_state=1)

    result = result.iloc[:, : ncolumns]
    result = result.infer_objects()
    for col in result:
        result[col] = result[col].fillna(-1)
    X = result["Country"]
    Y = result['Salary']
    del result["Country"]
    del result["Salary"]
    E = result.loc[:, ~result.columns.str.contains('^Unnamed')]

    return X, Y, E