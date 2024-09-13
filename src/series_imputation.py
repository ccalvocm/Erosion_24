# Nuevo Método de relleno mejorado
# Relleno diario por mes
import pandas as pd
import numpy as np
from scipy.stats import linregress

data = pd.read_csv("Tn.csv")
data = data.iloc[:, np.r_[0:4, np.where(data.apply(lambda x: x.count() > 5500))[0] + 4]]

var = "tmin"  # Variable a rellenar. Opciones: "tmax", "tmin", "pp", "q"

for g in range(1, 13):
    dat = data[data.iloc[:, 2] == g]
    dat0 = dat.copy()

    lnas = dat.iloc[:, 4:].isna().sum()
    posi = lnas.sort_values().index + 4

    dat2 = pd.concat([dat.iloc[:, :4], dat.iloc[:, posi]], axis=1)
    data2 = dat2.iloc[:, 4:]

    data3 = data2.copy()

    r = data2.corr(method="pearson")
    r2 = 0.7  # tolerancia m?nima para establecer el vector de relleno, temp = 0.8 y pp/q = 0.7

    for i in range(data3.shape[1]):
        res = pd.Series(index=["it_i", "it_j", "Rellenada", "Rellenadora", "Mes", "Falta", "Rellenado", "R2", "RMSE", "MAE", "Bias", "pval-KW", "pval-Flig"])
        dj = res.copy()
        for j in range(1, data3.shape[1]):
            eval = r.iloc[i, :].sort_values(ascending=False).values[j]
            if np.isnan(eval) or data2.iloc[:, i].isna().sum() == 0:
                continue
            elif eval >= np.sqrt(r2) and abs(eval) != 1:
                max_r = r.iloc[i, :].sort_values(ascending=False).index[j]
                pos = data2.columns.get_loc(max_r)

                m1 = linregress(data3.iloc[:, i], data2.iloc[:, pos])
                data2["est"] = m1.intercept + data2.iloc[:, pos] * m1.slope

                res = pd.Series([i, j, data2.columns[i], max_r, g, data2.iloc[:, i].isna().sum() - data2.iloc[:, -1].count(),
                                 data2.iloc[:, -1].count(), m1.rvalue**2,
                                 np.sqrt(((data2["est"] - data3.iloc[:, i])**2).sum() / data2["est"].notna().sum()),
                                 abs(data2["est"] - data3.iloc[:, i]).sum() / data2["est"].notna().sum(),
                                 (data2["est"] - data3.iloc[:, i]).sum() / data2["est"].notna().sum(),
                                 kruskal(data3.iloc[:, i], data2["est"]).pvalue,
                                 fligner(data2["est"], data3.iloc[:, i]).pvalue],
                                index=["it_i", "it_j", "Rellenada", "Rellenadora", "Mes", "Falta", "Rellenado", "R2", "RMSE", "MAE", "Bias", "pval-KW", "pval-Flig"])

                if var == "pp":
                    data2["est"] = np.where(data2["est"] < 0, 0, data2["est"])
                    data2["est"] = np.where(data2["est"] == m1.intercept, 0, data2["est"])
                else:
                    if var == "q":
                        pas = data.columns.get_loc(data3.columns[i])
                        value = data.loc[~data.iloc[:, pas].isna(), pas].groupby(data.iloc[:, 3]).min().mean()
                        data2.loc[data2["est"] < 0, "est"] = value
                    else:
                        data2["est"] = data2["est"]

                data2.iloc[:, i].fillna(data2.iloc[:, -1], inplace=True)

                if j == 1:
                    dj = res.copy()
                else:
                    dj = dj.append(res, ignore_index=True)

            else:
                continue

        if i == 0:
            dj2 = dj.copy()
        else:
            dj2 = dj2.append(dj, ignore_index=True)

    res_f = dj2.dropna(subset=["it_i"])
    res_f = res_f.drop_duplicates()

    if g == 1:
        DJ = res_f.copy()
    else:
        DJ = DJ.append(res_f, ignore_index=True)

    pcol = data2.columns.get_indexer(data.columns[posi])
    pcol = pcol[:-1]
    prow = data[data.iloc[:, 2] == g].index

    data.r.loc[prow, pcol] = data2.iloc[:, :-1].values

DJ = DJ.dropna(subset=["Rellenado"])

# Relleno 2da parte
dat = data.r.iloc[:, 1:]

dat_m = dat.groupby([dat.iloc[:, 0], dat.iloc[:, 1]]).apply(lambda x: np.sum(x) if x.count() > 24 else np.nan).reset_index()
dat_m.columns = dat.columns

dat_m1 = dat_m.copy()

# Remover outliers
for w in range(2, dat_m1.shape[1]):
    ploti = dat_m1.iloc[:, w].dropna()
    if len(ploti) == 0:
        continue
    else:
        aa = dat_m1[dat_m1.iloc[:, w].isin(ploti[ploti.apply(lambda x: x not in ploti.describe()["25%"] and x not in ploti.describe()["75%"])])].index
        dat_m1.iloc[aa, w] = np.nan

lnas = dat_m1.iloc[:, 2:].isna().sum()
posi = lnas.sort_values().index + 2

dat_m2 = pd.concat([dat_m1.iloc[:, :2], dat_m1.iloc[:, posi]], axis=1)

data = dat_m2.copy()

data2 = data.iloc[:, 2:]
pos = data2.columns[data2.apply(lambda x: x.count() >= 240)].values + 2
data = data.iloc[:, np.r_[0:3, pos]]

data2 = data.copy()
data3 = data2.copy()

r = data2.corr(method="pearson")
r2 = 0.7  # parametro de tolerancia de correlacion

for i in range(data2.shape[1]):
    for j in range(1, data2.shape[1]):
        eval = r.iloc[i, :].sort_values(ascending=False).values[j]
        if np.isnan(eval):
            continue
        elif eval >= np.sqrt(r2) and eval != 1:
            max_r = r.iloc[i, :].sort_values(ascending=False).index[j]
            pos = data2.columns.get_loc(max_r)

            m1 = linregress(data3.iloc[:, i], data2.iloc[:, pos])
            data2["est"] = m1.intercept + data2.iloc[:, pos] * m1.slope

            if var == "Pp":
                data2["est"] = np.where(data2["est"] < 0, 0, data2["est"])
                data2["est"] = np.where(data2["est"] == m1.intercept, 0, data2["est"])
            else:
                data2["est"] = data2["est"]

            data2.iloc[:, i].fillna(data2.iloc[:, -1], inplace=True)

            data2.iloc[:, i] = np.where(data2.iloc[:, i].isna(), data2.iloc[:, -1], data2.iloc[:, i])

    data2.iloc[:, i] = np.where(data2.iloc[:, i].isna(), data2.iloc[:, -1], data2.iloc[:, i])

exp = pd.concat([dat_m1.iloc[:, :2], data2.iloc[:, :-1]], axis=1)

# Segunda parte, relleno diario

data = data.r.iloc[:, 1:]
dat = exp.copy()

data = pd.concat([data.iloc[:, :3], data.loc[:, dat.columns[3:]].reindex(data.columns[3:], axis=1)], axis=1)

mn = dat.iloc[:, 0].unique()
yr = dat.iloc[:, 1].unique()

dd1 = data.groupby([data.iloc[:, 1], data.iloc[:, 2]]).mean().reset_index()
dd2 = data.groupby([data.iloc[:, 1], data.iloc[:, 2]]).std().reset_index()

dd3 = dat.groupby(dat.iloc[:, 1]).std().reset_index()

data2 = data.copy()

for i in range(3, dat.shape[1]):
    for j in range(len(yr)):
        for k in range(len(mn)):
            value = dat[(dat.iloc[:, 0] == mn[k]) & (dat.iloc[:, 1] == yr[j])].iloc[:, i].values[0]
            vec = dd1[(dd1.iloc[:, 0] == mn[k])].index
            posi = (np.abs(value - dd1.iloc[vec, i])).idxmin()
            mes = dd1.iloc[posi, 0]
            ano = dd1.iloc[posi, 1]

            dat_e = data[(data.iloc[:, 1] == mes) & (data.iloc[:, 2] == ano)].iloc[:, i + 1]
            if dat_e.isna().sum() > 0:
                dat_e[dat_e.isna()] = dat_e.mean()
            else:
                dat_e = dat_e

            newvec = value + (dat_e - dat_e.mean()) if var in ["tmin", "tmax"] else dat_e / (dat_e.sum() / value)
            newvec = np.round(newvec, 2)

            if len(data2[(data.iloc[:, 1] == mn[k]) & (data.iloc[:, 2] == yr[j])].iloc[:, i + 1]) == len(newvec):
                data2.loc[(data.iloc[:, 1] == mn[k]) & (data.iloc[:, 2] == yr[j]), data.columns[i + 1]] = newvec
            else:
                veq = np.concatenate([newvec, np.repeat(np.mean(newvec), len(data2[(data.iloc[:, 1] == mn[k]) & (data.iloc[:, 2] == yr[j])].iloc[:, i + 1]))])
                pot = len(data2[(data.iloc[:, 1] == mn[k]) & (data.iloc[:, 2] == yr[j])].iloc[:, i + 1])
                data2.loc[(data.iloc[:, 1] == mn[k]) & (data.iloc[:, 2] == yr[j]), data.columns[i + 1]] = veq[:pot]

data3 = data.copy()

for i in range(3, data.shape[1]):
    data3.iloc[:, i] = np.where(data.iloc[:, i].isna(), data2.iloc[:, i], data.iloc[:, i])
    if var in ["pp"]:
        ss = pd.DataFrame(data3.iloc[:, i].value_counts().sort_values(), columns=["count"])
        ss2 = ss[ss["count"].astype(str).str.len() > 5]
        ss3 = ss2[ss2["count"] > ss2["count"].mean()]
        posi = data3.iloc[:, i].isin(ss3[ss3["count"].astype(float) < 5].index)
        data3.loc[posi, data.columns[i]] = 0

data3.to_csv("E:/Rellenas_tn_prueba_GY.csv", index=False)
