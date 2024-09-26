import series_imputation as si
import os

def benchmark():
    # TO DO
    return None

def main():
    data_path = os.path.join('..','data','precipitation',
'gauges_data','est_DMC_2024-05-23.xlsx')
    data=si.rusleData(data_path,20)
    data.loadData()
    data.loadGaugesCoordinates()
    data.min_years()
    pp_fill = data.fillData()
    import matplotlib.pyplot as plt
    data.pp.plot(legend=False)
    pp_fill.plot(legend=False)

if __name__ == '__main__':
    main()
