import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

FILE_PATH = './Rocks.xls'
CLASSES = ['granite', 'diorite', 'marble', 'slate', 'limestone', 'breccia']
COLUMS = ['RMCS', 'RMFX', 'AAPN']
ALPHA = 0.05


def load_data():
    df = pd.read_excel('Rocks.xlsx', sheet_name='Data', usecols=['Code', 'Class', 'RMCS', 'RMFX', 'AAPN'])
    return df

def create_lists(df: pd.DataFrame):
    
    graph = list()
    datas = list()
    for b in CLASSES:
        data = df[df['Class'] == b]
        graph.append(data)
        for a in COLUMS:
            class_dict = {
                'Class' : b,
                'Colums' : a,
                'Count' : len(data),
                'Mean' : np.mean(data[a]),
                'Variance' : np.var(data[a]),
                'Stand_Error': (np.std(data[a]) / np.sqrt(len(data[a]))),
                'Q1': np.percentile(data[a], 25),
                'Q3': np.percentile(data[a], 75),
                'normality': True if st.kstest(data[a], 'norm')[1] < 0.05 else False
            }
            datas.append(class_dict)
    return datas, graph

def draw_boxPlots(graph,key):


    data_1 = graph[0][key]
    data_2 = graph[1][key]
    data_3 = graph[2][key]
    data_4 = graph[3][key]
    data_5 = graph[4][key]
    data_6 = graph[5][key]
    data = [data_1, data_2, data_3, data_4,data_5,data_6]

    plt.figure(figsize=(10, 7))
    plt.title(key)


    # Creating plot
    plt.boxplot(data, labels=CLASSES)

    # show plot
    plt.show()
    
def print_info(datas:list()):
    for item in datas:
        print("Základní statistické informace o datové sadě: Třída: {}, Sloupec: {}, EX: {}, VAR: {}, NORMALIT: {}".format(item['Class'], item['Colums'], item['Mean'], item['Variance'],item['normality']))
    
def T_Tests(datas: list()):
    print("Pouze porovnán kde nedošlo k zamítnutí hypotézi")
    for i in range(len(datas)):
        k1 = datas[i]
        for j in range(i + 1, len(datas), 1):
            k2 = datas[j]
            for key in k1['Class']:
                key1 = key
                break
            for key in k2['Class']:
                key2 = key
                break
            
            if(key1 != key2):
                for item in COLUMS:
                    if(t_testCompare(k1=k1, k2=k2, param=item)):
                        print("podle 1D t-testu {} jsou  {} a {} stejné".format(item, str(key1), str(key2)))
                        
                if(t_testCompare2D(k1,k2,COLUMS[0],COLUMS[1],key1,key2)):
                    print("podle 2D t-testu {} a {} jsou  {} a {} stejné".format(COLUMS[0], COLUMS[1], str(key1), str(key2)))

                if (t_testCompare2D(k1, k2, COLUMS[0],COLUMS[2],key1,key2)):
                   print("podle 2D t-testu {} a {} jsou  {} a {} stejné".format(COLUMS[0], COLUMS[2], str(key1), str(key2)))

                if (t_testCompare2D(k1, k2, COLUMS[1], COLUMS[2],key1,key2)):
                    print("podle 2D t-testu {} a {} jsou  {} a {} stejné".format(COLUMS[1], COLUMS[2], str(key1), str(key2)))
                    
                if((t_testCompare3d(k1,k2,COLUMS[0],COLUMS[1],COLUMS[2]))):
                    print("podle 3D t-testu všech sloupců jsou  {} a {} stejné".format(str(key1), str(key2)))
                
def t_testCompare(k1, k2, param):
    p_val=st.ttest_ind(k1[param], k2[param]).pvalue
    if p_val<ALPHA:
        return False
    else:
        return True
    

def t_testCompare2D(k1, k2, param, param1,key1,key2):
    tmp11 = []
    for i in k1[param]:
        tmp11.append(i)
    tmp12 = []
    for i in k1[param1]:
        tmp12.append(i)

    tmpK1 = []
    for i in range(len(tmp11)):
        tmpK1.append([tmp11[i],tmp12[i]])

    tmp21 = []
    for i in k2[param]:
        tmp21.append(i)
    tmp22 = []
    for i in k2[param1]:
        tmp22.append(i)

    tmpK2 = []
    for i in  range(len(tmp21)):
        tmpK2.append([tmp21[i], tmp22[i]])


    plt.scatter(tmp11, tmp12, color='Blue')
    plt.scatter(tmp21, tmp22, color='Red')
    plt.xlabel(param)
    plt.ylabel(param1)
    p_val = st.ttest_ind(tmpK1, tmpK2).pvalue

    if (p_val[0] > ALPHA) and (p_val[1] > ALPHA):
        plt.legend(["Modré: " + str(key1), "Červené: " + str(key2)])
        plt.show()
        return True
    else:
        plt.cla()
        return False
    
def t_testCompare3d(k1, k2, param, param1, param2):
    tmp1 = []
    for i in k1[param]:
        tmp1.append(i)
    tmp2 = []
    for i in k1[param1]:
        tmp2.append(i)

    tmp3 = []
    for i in k1[param2]:
        tmp3.append(i)

    tmpK1 = []
    for i in range(len(tmp1)):
        tmpK1.append([tmp1[i], tmp2[i],tmp3[i]])

    tmp1 = []
    for i in k2[param]:
        tmp1.append(i)
    tmp2 = []
    for i in k2[param1]:
        tmp2.append(i)
    tmp3 = []
    for i in k2[param2]:
        tmp3.append(i)

    tmpK2 = []
    for i in range(len(tmp1)):
        tmpK2.append([tmp1[i], tmp2[i], tmp3[i]])

    p_val = st.ttest_ind(tmpK1, tmpK2).pvalue

    if (p_val[0] > ALPHA) and (p_val[1] > ALPHA) and (p_val[2] > ALPHA):
        return True
    else:
        return False
    
def F_tests(datas: list):
    
    for i in range(len(datas)):
        k1 = datas[i]
        for j in range(i + 1, len(datas), 1):
            k2 = datas[j]
            for key in k1['Class']:
                key1 = key
                break
            for key in k2['Class']:
                key2 = key
                break
            
            if(key1 != key2):
                for item in COLUMS:
                    if(f_test_1D(k1=k1, k2=k2, param=item)):
                        print("podle 1D F-testu {} jsou  {} a {} stejné".format(item, str(key1), str(key2)))
                  
                if(f_testCompare2D(k1,k2,COLUMS[0],COLUMS[1])):
                    print("podle 2D f-testu {} a {} jsou  {} a {} stejné".format(COLUMS[0], COLUMS[1], str(key1), str(key2)))

                if (f_testCompare2D(k1, k2, COLUMS[0],COLUMS[2])):
                   print("podle 2D f-testu {} a {} jsou  {} a {} stejné".format(COLUMS[0], COLUMS[2], str(key1), str(key2)))

                if (f_testCompare2D(k1, k2, COLUMS[1], COLUMS[2])):
                    print("podle 2D f-testu {} a {} jsou  {} a {} stejné".format(COLUMS[1], COLUMS[2], str(key1), str(key2)))
                    
                if((t_testCompare3d(k1,k2,COLUMS[0],COLUMS[1],COLUMS[2]))):
                    print("podle 3D F-testu všech sloupců jsou  {} a {} stejné".format(str(key1), str(key2)))   
def f_test_1D(k1, k2, param):
    f_value = (np.var(k1[param])/ np.var(k2[param]))
    fp_value = st.f.cdf(f_value, len(k1[param]) - 1, len(k2[param]) - 1)
    if fp_value > ALPHA:
        return True
    else:
        return False
    
def f_testCompare2D(k1, k2, param, param1):
    
    if(f_test_1D(k1=k1, k2=k2,param=param) and f_test_1D(k1=k1, k2=k2,param=param1)):
        return True
    else:
        return False
    
def f_testCompare3d(k1, k2, param, param1, param2):
    
    if(f_test_1D(k1=k1, k2=k2,param=param) and f_test_1D(k1=k1, k2=k2,param=param1) and f_test_1D(k1=k1, k2=k2,param=param2)):
        return True
    else:
        return False
    """if fp_value[0] > ALPHA and fp_value[1] > ALPHA:
        return True
    else:
        return False
def create_dicitinary(df: pd.DataFrame):
    rocks = {}
    rocks_classes = df["Class"]
    for i in range(len(rocks_classes)):
        class_rock = rocks_classes[i]
        class_rock = str.replace(class_rock, " ", "")
        if class_rock not in rocks:
            rocks[class_rock] = {k:[] for k in df}
        for k in df:
            rocks[class_rock][k].append(df[k][i])
            
    return rocks

def basic_statistic(rocks: dict):
    statistics = {}
    
    #Základní statistická analáza
    keys = []
    for k in rocks:# Průchod pře všechny třídy
        keys.append(k)
        statistics[k] = {"EX":{}, "varX":{}, "std":{}, "stand_error":{}, "Q1":{}, "Q3":{}, "normality":{}}
        for f in COLUMS:
            statistics[k]["EX"][f] = np.mean(rocks[k][f])
            statistics[k]["varX"][f] = np.var(rocks[k][f])
            statistics[k]["std"][f] = np.std(rocks[k][f])
            statistics[k]["stand_error"][f] = (np.std(rocks[k][f]/np.sqrt(np.size(rocks[k][f]))))
            statistics[k]["Q1"][f] = np.percentile(rocks[k][f], 25)
            statistics[k]["Q3"][f] = np.percentile(rocks[k][f], 75)
            statistics[k]["normality"][f] = True if stats.kstest(rocks[k][f], 'norm')[1] < ALPHA else False 
    print("Test normality pro třídu a sloupec", {k:statistics[k]["normality"] for k in keys})
    
    return keys, statistics
    
def N_dimension(keys, rocks):
    
    t_tests = {}
    f_tests = {}

    anova = {}
    print(keys)
    for f in COLUMS:# průchod přes více dimenzí
        f_stones = [rocks[k][f] for k in keys]
        plt.boxplot(f_stones, labels=keys)
        plt.title(f)
        plt.show()
        t_tests[f] = []
        f_tests[f] = []
        for i in range(len(keys)):
            k1 = keys[i]
            for j in range(i+1, len(keys), 1):
                k2 = keys[j]
                rocks1 = rocks[k1][f]
                rocks2 = rocks[k2][f]
                tp_value = stats.ttest_ind(rocks1, rocks2).pvalue
                tr = True# Hypotéza nastavena na true
                if tp_value <  ALPHA:#Pokud je hodnota menšínež nastavená alfa je Hypotéza zamítnuta
                    tr = False
                t_tests[f].append((k1+" - "+k2, tr))

    return t_tests"""
            
if __name__ == '__main__':
    
    df = load_data()
    datas, graph = create_lists(df=df)
    print_info(datas=datas)
    for item in COLUMS:
        draw_boxPlots(graph=graph, key=item)
    #T_Tests(graph)
    F_tests(graph)
    """keys, statistic = basic_statistic(rocks=data)
    t_test = N_dimension(keys=keys, rocks=data) #Zde probíhá párová test hypotéz
    for k, v in t_test.items():
        for a in v:
            print("Parová testování pro sloupec {} a par ({}): {}".format(k , a[0], a[1])) #Zde je výpis získaného výsledku a je zde řečeno, zda se jedná o zamítnutí či né """
    
    
            
    