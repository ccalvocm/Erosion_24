#%% Dependencias

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from itertools import cycle
import geopandas as gpd
import os
import re
import matplotlib.pyplot as plt
import datetime
import pathlib
#Funciones

def mejoresCorrelaciones(df, col, Nestaciones):
    ordenados = df.copy().sort_values(by=col, ascending = False)
    # coef. correlacion pearson 0.5
    ordenados = ordenados[ordenados[col] >= 0.7]
    return ordenados.index

def parse_digito_verificador(lista):
    list_return=[]
    for rut in lista:
        rut=str(rut)
        if len(rut)<=7:
            rut='0'+rut
        digito_ver=digito_verificador(rut)
        list_return.append(str(rut)+'-'+str(digito_ver))
    return list_return

# Función rut
def digito_verificador(rut):
    reversed_digits = map(int, reversed(str(rut)))
    factors = cycle(range(2, 8))
    s = sum(d * f for d, f in zip(reversed_digits, factors))
    if (-s) % 11 > 9:
        return 'K'
    else:
        return (-s) % 11

def min_dist(point, gpd2, n_multivariables):
    gpd2['Dist'] = gpd2.apply(lambda row:  point.distance(row.geometry),axis=1)
    gpd2=gpd2[gpd2['Dist']<=5.3e4]
    return gpd2.sort_values(by=['Dist']).loc[gpd2.sort_values(by=['Dist']).index[0:n_multivariables],
                                             gpd2.columns]

def join_path(*args):
    string=os.path.abspath('\\'.join(args))         
    return string
    
def parse_att_fisicos(df):
    atts=['mean_elev','mean_slope_perc','_forest','_grass','shrub_frac',
'geol_class_1st_','crop_frac','land_cover_missing','frac_snow']
    atts_fisico=df.columns[df.columns.str.contains('|'.join(atts))]
    atts_fisico=[x for x in atts_fisico if ('frac_snow_tmpa' not in x)]
    return df[atts_fisico]

def similitud_fisica(est,df_fisic_norm,df_coord_reg,df_coord_camels):
    '''
    Quevedo et al. (2021)
    1. Se elabora un ranking de similitud con las estaciones vecinas, 
    considerando el set de atributos asociado al criterio de interés y la 
    ecuación 2.1. la fórmula de similitud es Sij=Sum(ai-aj)/IQR(a)
    2. Se calcula el coeficiente de determinación entre los caudales de la cuenca
    objetivo y los caudales de las diez cuencas más similares según el ranking 
    efectuado anteriormente.
    3. De las diez cuencas encontradas, se escoge un máximo de cuatro cuencas 
    candidatas que cuenten con coeficientes de determinación mayor a 0,5.
    4. Se ajusta un modelo lineal entre la estación objetivo y la primera 
    estación candidata, y se rellenan los caudales faltantes con dicho modelo. 
    Se repite el procedimiento con las estaciones candidatas restantes 
    (considerando sólo datos observados) y se estiman los datos que no pudieron
    ser rellenados con estaciones de mejor ranking.
    '''      
    # filtrar primero las estaciones a menos de 200km
    coord_est=df_coord_reg[df_coord_reg['rut']==est]
    est_near=min_dist(coord_est.geometry,df_coord_camels, 200)
    idx_near=est_near.index.intersection(df_fisic_norm.index)
    df_fisic_norm_near=df_fisic_norm.loc[idx_near].copy()
    
    # evaluar la similitud fisica de otras cuencas
    try:
        row=df_fisic_norm_near.loc[est]
        Sij=np.abs(df_fisic_norm_near-row)/(df_fisic_norm_near.quantile(.75)-df_fisic_norm_near.quantile(.25))
        Sij_rms=Sij.apply(lambda x: np.sqrt(np.sum(x**2)),axis=1)
        est_simil=Sij_rms.sort_values(ascending=True)[1:22]
        return est_simil
    except:
        return pd.DataFrame(index=[est])

def min_years(df_mon,minYr):
   
   data=df_mon.notnull().astype('int')
   data=data.groupby(df_mon.index.year)  
   data_anual=data.aggregate(np.sum)
   data_anual=data_anual/(12*0.8)  
   data_anual = data_anual.apply(lambda x: [y if y < 1 else 1 for y in x])
   data_anual = data_anual.transpose()
  
   data_anual = data_anual.sort_index()
   estaciones_minimas=pd.DataFrame(data_anual.sum(axis=1),columns=['registro'])
   estaciones_minimas=estaciones_minimas[estaciones_minimas['registro']>=minYr]
   
   return estaciones_minimas

def loadCAMELS():
# leer dataset CAMELS
    path_camels=os.path.join('..','data','CAMELS_CL_v202201')
    camels=pd.read_csv(os.path.join(path_camels,'catchment_attributes.csv'),
                    index_col=0)
    return camels

def remove_duplicates(df):
    # leer y completar estaciones duplicadas
    return df.drop_duplicates()

def verificationDigit(camels):
    # calcular el digito verificador
    camels.index=parse_digito_verificador(camels.index)
    return camels
    
    # atributos fisicos
    camels_fisico=parse_att_fisicos(camels)

    # normalizar cada columna de los atributos fisicos
    camels_fisic_norm=camels_fisico.apply(lambda x: np.abs((x-x.mean())/x.std()),
                                                        axis=0)
    camels_fisic_norm=camels_fisic_norm.apply(lambda x: x/x.max(),axis=0)

    return camels_fisic_norm

def camelsCoords(camels):
    # coordenadas de las estaciones CAMELS
    # coordenadas camels
    coords_camels=gpd.GeoDataFrame(camels,
    geometry=gpd.points_from_xy(x=camels['gauge_lon'],y=camels['gauge_lat']))
    coords_camels.set_crs(epsg='4326',inplace=True)
    coords_camels.to_crs(epsg='32719',inplace=True)
    return coords_camels

def loadData(path):
# leer precipitacion
    pp=pd.read_excel(path,sheet_name='Datos',index_col=0,
                            parse_dates=True,skiprows=[1])
    return pp
    
# leer metadata dga de Maule y Biobio (Nuble esta incluida)
def loadGaugesCoordinates(data_path):
    metadata=pd.read_excel(data_path,sheet_name='Fichas',index_col=0)

    metadata=remove_duplicates(metadata)
    # transpose
    metadata=metadata.transpose()

    # gdf de metadata
    gdf_metadata=gpd.GeoDataFrame(metadata,
            geometry=gpd.points_from_xy(x=metadata['Longitud'],
            y=metadata['Latitud']))
    gdf_metadata.set_crs(epsg='4326',inplace=True)
    gdf_metadata.to_crs(epsg='32719',inplace=True)
    return gdf_metadata

def main(root='.',data_path=join_path(root,'..','data','precipitation',
'gauges_data','est_DMC_2024-05-23.xlsx'),yr_ini=1980,minYr = 20,
n_multivariables=4,stdOutliers=3):
    """
    

    Parameters
    ----------
    root : str
        carpeta de trabajo, ejemplo r'G:\OneDrive - ciren.cl\2022_Nuble_Embalses'
    cuenca : str
        cuenca o region de análisis.
    yr_ini : str o int
        año de inicio desde el cual se realizará el relleno y extensión de data.
    path_q_0 : str
        ruta de la planilla de caudales de region o cuenca al norte del área 
        en estudio. Ejemplo join_path(root,'Datos','Caudales',
                                'CaudalesDGA_Maule_2021_revA.xlsx')
    path_q_region : str
        ruta de la planilla de caudales de region o cuenca en estudio. Ejemplo
  ruta de la planilla de caudales de region o cuenca al norte del área 
  en estudio. Ejemplo join_path(root,'Datos','Caudales',
                           'CaudalesDGA_Nuble_2021_revA.xlsx')
    path_q_2 : str
        ruta de la planilla de caudales de region o cuenca al sur del área 
        en estudio. Ejemplo join_path(root,'Datos','Caudales',
                                'CaudalesDGA_BioBio_2021_revA.xlsx')
    ruta_reg : str
        ruta del shape de la cuenca o región en estudio. Ejemplo
        os.path.join('..','SIG','REGION_NUBLE','region_Nuble.shp')
    path_shac : str
        ruta de los SHACS a nivel nacional. Ejemplo 
os.path.join('..', 'SIG', 'SHACS',
                          'Acuiferos_SHAC_Julio_2022.shp')
    
    Notas: Necesariamente debe existir una carpeta con el dataset de las cuencas 
        CAMELS y debe ubicarse en la siguiente ruta:
    os.path.join('..','Datos','dataset_cuencas','CAMELS_CL_v202201')      


    Returns
    -------
    None.
    
    Outputs
    -------
    Caudales medios mensuales rellenados y extendidos.

    """
#%%    

    # leer dataset CAMELS
    camels=loadCAMELS()
    # calcular el digito verificador
    camels=verificationDigit(camels)
    # atributos fisicos
    camels_fisico=parse_att_fisicos(camels)
    # normalizar cada columna de los atributos fisicos
    camels_fisic_norm=camels_fisico.apply(lambda x: np.abs((x-x.mean())/x.std()),
                                                        axis=0)
    camels_fisic_norm=camels_fisic_norm.apply(lambda x: x/x.max(),axis=0)
    # coordenadas de las estaciones CAMELS
    # coordenadas camels
    coords_camels=camelsCoords(camels)
    # leer metadata dga de Maule y Biobio (Nuble esta incluida)
    metadata=loadGaugesCoordinates(data_path)
    # filtrar estaciones de la región
    gdf_metadata=filter_stations(metadata)
    # leer precipitacion
    pp=loadData(data_path)
    # leer y completar estaciones duplicadas
    estaciones=remove_duplicates(pp.columns)
    # leer dataset CAMELS
    camels=loadCAMELS()
    # calcular el digito verificador

    # rango de fechas de las nuevas normales
    date_start=pd.to_datetime(str(yr_ini)+'-04-01')
    date_end=pd.to_datetime(str(datetime.date.today().year)+'-03-31')

    # seleccionar estaciones con un minimo de 20 years (Quevedo, 2021)
    estaciones_min=min_years(pp,minYr)
    pp_filtradas=pp.copy()[estaciones_min.index]

    def missingDataInput():
        # TODO
        return None

    def missingData(pp_filtradas,metadata,coords_camels,gdf_metadata,camels_fisic_norm,n_multivariables,stdOutliers):
        # meses
        meses=range(1,13)
        
        # df relleno
        q_mon_MLR=pp_filtradas.copy()
        q_mon_MLR=q_mon_MLR.astype(float)
        
        # Multivariable
        # rellenar estaciones en shacs
        estaciones=pp_filtradas.columns.intersection(est_utm_shac['rut'])
        
        df_check=q_mon_MLR[estaciones].copy()
        
        while df_check.isnull().values.any():
            for col in estaciones:
                # print('Rellenando estación '+str(col),metadata[metadata['rut']==col]['Estacion'].iloc[0])
                            
                for mes in meses:
                    q_mon_mes=pp_filtradas.loc[pp_filtradas.index.month==mes].copy()
                    y=q_mon_mes[col]
                            
                    if y.count() < 1:
                        continue
                    
                    # obtener estaciones con similitud fisica
                    # pueden haber estaciones fuera de los shacs segun Quevedo, 2021
                    est_similares=similitud_fisica(col,camels_fisic_norm,
                                                gdf_metadata,coords_camels).index         
        
                    # similitud hidrológica
                    correl=q_mon_mes.astype(float).corr()
                    coord_est=gdf_metadata[gdf_metadata['rut']==col].geometry
                    est_near=min_dist(coord_est.geometry,gdf_metadata, -1)
                    idx=q_mon_mes.columns.intersection(list(est_near['rut']))
                    est_indep=mejoresCorrelaciones(correl.loc[list(idx)],col, -1)

                    if len(est_similares)>1:
                        est_indep=[col]+list(set(est_indep) & set(est_similares))
                    # a lo más 4 estaciones para rellenar
                    est_indep=list(est_indep[:n_multivariables])+[col]
                    est_indep=list(set(est_indep))
                    if col in ['04522001-K','04531001-9','04535002-9','04506002-0','04550003-9']:
                        print(metadata[metadata['rut'].isin(est_indep)]['Estacion'].values)
                    x=pd.DataFrame(q_mon_mes.loc[q_mon_mes.index.month==mes][est_indep].copy(),
                                                                    dtype=float)
                    
                    x=x.dropna(how='all',axis=1)
                    
                    max_value_=x.mean()+stdOutliers*x.std()
                    
                    imp=IterativeImputer(imputation_order='descending',random_state=0,
                max_iter=20,min_value=0,max_value=max_value_,sample_posterior=False,initial_strategy='median',skip_complete=True)
                    Y=imp.fit_transform(x)
                    Q_monthly_MLR_mes=pd.DataFrame(Y,columns=x.columns,index=x.index)
                    Q_monthly_MLR_mes=Q_monthly_MLR_mes.dropna()
        
                    q_mon_MLR.loc[Q_monthly_MLR_mes.index,
                                col]=Q_monthly_MLR_mes[col].values
                    df_check=q_mon_MLR[estaciones].copy()
        return q_mon_MLR

        
