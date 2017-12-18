import pandas as pd
import lectura
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import r2_score


#-----------------------------------------------------------------------------
# Etapa 2

#file = "tabla_calibration_validation_conSMSMAP_2.csv"
#file = "tabla_completa_sigma5km_GPM.csv"
#file = "tabla_completa_sigma5km_GPM_Et_HR.csv"
file = "tabla_completa_sigma5km_GPM_HR_Et.csv"


data = lectura.lecturaSimple_etapa2(file)
data = data.rename(index=str, columns={"Et":"ET"})




#-----------------------------------------------------------------------------
### Etapa 1
#file = "tabla_calibration_validation.csv"
#data = lectura.lecturaSimple_etapa1(file)

data = data.rename(index=str, columns={"Sigma0":"$\sigma^0$"})
data = data.rename(index=str, columns={"RSOILTEMPC":"Ts"})
data = data.rename(index=str, columns={"T_aire":"Ta"})
data = data.rename(index=str, columns={"Tension_va":"e_a"})
#data = data.rename(index=str, columns={"SM_CONAE":"HS_CONAE"})


data = data.rename(index=str, columns={"SM_SMAP":"HS_SMAP"})
data = data.rename(index=str, columns={"T_s":"Ts"})


data =data[["HS_SMAP","ET","HR","$\sigma^0$", "Ts", "PP"]]
print data

#del data['T_a']
data.PP = data.PP*0.1
#data.Et = data.Et*0.1/8.0




axes = scatter_matrix(data, alpha=0.4, figsize=(6, 6), diagonal='hist')
print type(axes)

#RR = r2_score(axes[i, :], axes[:, j])

corr = data.corr().as_matrix()
print corr
for i, j in zip(*plt.np.triu_indices_from(axes, k=1)):
    axes[i, j].cla()
    axes[i, j].annotate("r=%.3f" %corr[i,j], (0.5, 0.5), xycoords='axes fraction', ha='center', va='center')
plt.show()

