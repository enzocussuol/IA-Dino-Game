import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

resultados_professor = [1214.0, 759.5, 1164.25, 977.25, 1201.0, 930.0, 1427.75, 799.5, 
            1006.25, 783.5, 728.5, 419.25, 1389.5, 730.0, 1306.25, 675.5, 1359.5, 1000.25, 1284.5, 1350.0, 751.0, 
            1418.75, 1276.5, 1645.75, 860.0, 745.5, 1426.25, 783.5, 1149.75, 1482.25]

meus_resultados = [2100.0, 2249.25, 1932.25, 1869.75, 1819.0, 2389.75, 1591.5, 1925.5, 
1985.5, 2067.25, 2205.0, 1669.0, 1727.75, 1948.0, 1440.25, 2156.0, 2208.5, 2010.75, 
1932.5, 2244.0, 1596.75, 2301.75, 1799.5, 2263.75, 1401.0, 1627.0, 2321.25, 1824.0, 2056.0, 1876.0]

scores = {'Resultados do Professor': resultados_professor,
        'Meus Resultados': meus_resultados}

df = pd.DataFrame(scores)
output = sns.boxplot(data=df)

plt.savefig('boxplot.png')

tabelaPareada = [[0 for x in range(2)] for y in range(2)]

tabelaPareada[0][0] = ('Resultados do Professor', resultados_professor)
tabelaPareada[1][1] = ('Meus Resultados', meus_resultados)

for i in range(0, 2):
    for j in range(0, 2):
        if i != j:
            if i < j:
                s, p = ttest_rel(tabelaPareada[i][i][1], 
                                 tabelaPareada[j][j][1])
            if i > j:
                s, p = wilcoxon(tabelaPareada[i][i][1], 
                                tabelaPareada[j][j][1])
            tabelaPareada[i][j] = p

for i in range(0, 2):
    for j in range(0, 2):
        if i == j:
            print(tabelaPareada[i][j][0], " ", end="")
        else:
            print("%0.10f" % tabelaPareada[i][j], " ", end="")
    print("\n")