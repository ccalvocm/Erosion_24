import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from itertools import cycle
import geopandas as gpd
import os

class rusleData(object):
    def __init__(self, data_path=os.path.join('..','data','precipitation',
'gauges_data','est_DMC_2024-05-23.xlsx'),minYr = 20):
        self.data_path = data_path
        self.minYr = minYr

    def loadData(self):
        # read precipitation data
        self.pp=pd.read_excel(self.data_path,sheet_name='Datos',index_col=0,
                                parse_dates=True,skiprows=[1])
        return None
    
    def loadGaugesCoordinates(self):
        # read gauges metadata for coordinates
        metadata=pd.read_excel(self.data_path,sheet_name='Fichas',
                               index_col=0)

        metadata=metadata.drop_duplicates()
        # transpose
        metadata=metadata.transpose()

        metadata=metadata.set_index(metadata.columns[0],drop=True)

        # metadata gdf
        gdf_metadata=gpd.GeoDataFrame(metadata,
                geometry=gpd.points_from_xy(x=metadata['Longitud'],
                y=metadata['Latitud']))
        
        # convert to UTM coordinates to calculate distances
        gdf_metadata.set_crs(epsg='4326',inplace=True)
        gdf_metadata.to_crs(epsg='32719',inplace=True)
        self.metadata = gdf_metadata
        return None
    
    def min_years(self):
        #    filter minimum number of years with data
        data=self.pp.copy().notnull().astype('int')
        data=data.groupby(data.index.year)  
        data_anual=data.aggregate(np.sum)
        data_anual=data_anual/(12*0.8)  
        data_anual = data_anual.apply(lambda x: [y if y < 1 else 1 for y in x])
        data_anual = data_anual.transpose()
        data_anual = data_anual.sort_index()
        est_min=pd.DataFrame(data_anual.sum(axis=1),columns=['registro'])
        est_min=est_min[est_min['registro']>=self.minYr]
        return est_min

    def trendAnalysis():
        # TO DO
        return None

    def missingDataImputer(self,n_multivariables=4,stdOutliers=3):

        def min_dist(point, gpd2, n_multivariables):
            gpd2['Dist'] = gpd2.apply(lambda row:  point.distance(row.geometry),axis=1)
            gpd2=gpd2[gpd2['Dist']<=5.3e4]
            return gpd2.sort_values(by=['Dist']).loc[gpd2.sort_values(by=['Dist']).index[0:n_multivariables],
                                                    gpd2.columns]
        # Functions
        def bestCorrelations(df, col):
            ordered = df.copy().sort_values(by=col, ascending = False)
            # Pearson correlation coefficient 0.7
            ordered = ordered[ordered[col] >= 0.7]
            return ordered.index
        
        # months
        months=range(1,13)
        
        # filled dataframe
        q_mon_MLR=self.pp.copy().astype(float)

        for col in q_mon_MLR.columns:
                        
            for mon in months:
                q_mon_m=self.pp.loc[self.pp.index.month==mon].copy()
                y=q_mon_m[col]
                        
                if y.count() < 1:
                    continue
                
                #similarity
                correl=q_mon_m.astype(float).corr()
                coord_est=self.metadata.loc[col].geometry
                est_near=min_dist(coord_est,self.metadata, -1)
                idx=q_mon_m.columns.intersection(list(est_near.index))
                est_indep=bestCorrelations(correl.loc[list(idx)],col)

                # at most 4 stations for filling
                est_indep=list(est_indep[:n_multivariables])+[col]
                est_indep=list(set(est_indep))
                x=pd.DataFrame(q_mon_m.loc[q_mon_m.index.month==mon][est_indep].copy(),
                                                                dtype=float)
                
                x=x.dropna(how='all',axis=1)
                
                max_value_=x.mean()+stdOutliers*x.std()
                
                imp=IterativeImputer(imputation_order='descending',random_state=0,
            max_iter=20,min_value=0,max_value=max_value_,sample_posterior=False,
            initial_strategy='median',skip_complete=True)
                Y=imp.fit_transform(x)
                Q_monthly_MLR_mes=pd.DataFrame(Y,columns=x.columns,index=x.index)
                Q_monthly_MLR_mes=Q_monthly_MLR_mes.dropna()

                q_mon_MLR.loc[Q_monthly_MLR_mes.index,
                            col]=Q_monthly_MLR_mes[col].values
        return q_mon_MLR

    def fillData(self):
        est_min = self.min_years()
        self.pp = self.pp.copy()[est_min.index]
        self.pp_fill = self.missingDataImputer()
        return self.pp_fill
    
def main():
    return None

if __name__ == '__main__':
    main()
