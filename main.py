import pandas as pd
# import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import missingno as msno
from pandasql import sqldf
from scipy.stats import mstats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import linear_model, neighbors, tree, svm, ensemble
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
# from scipy.stats import boxcox
# from scipy.special import inv_boxcox
q = lambda q: sqldf(q, globals())

### FUNCTIONS - ETL
def do_etl():
    df = pd.read_csv("source_data/historical_product_demand.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = sqldf("""SELECT product_code
                       || "-"
                       || warehouse
                       || "-"
                       || product_category AS model_key,
                       date,
                       Sum(IFNULL(order_demand, 0))   AS Order_Demand
                FROM   df
                GROUP  BY 1,
                          2 """, locals())
    df = sqldf("""SELECT a.*
                FROM   df a
                       INNER JOIN (SELECT model_key,
                                          Count(*) n
                                   FROM   df
                                   GROUP  BY 1
                                   ORDER  BY 2 DESC,
                                             1
                                   LIMIT  30) b
                               ON b.model_key = a.model_key
                ORDER  BY model_key,
                          Date """, locals())
    df.to_csv("tables/1_thirty_models.csv", index=False)

### FUNCTIONS - Exploration
def do_exploration():
    df = pd.read_csv("tables/1_thirty_models.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index(["Date"])
    thirty_plots(df=df, filename="1_original_data")
    # Good min/max dates: 1/5/12 and 12/28/16
    # Will cap at 95% upper bound and 5% lower bound
    # Will not log the data
    # Will impute using 7 day moving average
    # There are lots of missing values
    print("There are infinity values: {}".format(df.fillna(0).replace([np.inf, -np.inf], np.nan).isnull().values.any()))
    # There are no infinity values

def thirty_plots(df, filename):
    df = pd.pivot_table(data=df, values='Order_Demand', index="Date", columns="model_key")
    df.to_csv("tables/_row_date__col_product.csv")
    colors=['#A8E6CE', '#A8E6CE', '#A8E6CE',
            '#DCEDC2', '#DCEDC2', '#DCEDC2',
            '#FFD3B5', '#FFD3B5', '#FFD3B5',
            '#FFAAA6', '#FFAAA6', '#FFAAA6',
            '#FF8C94',  '#FF8C94', '#FF8C94',
            '#A8E6CE', '#A8E6CE', '#A8E6CE',
            '#DCEDC2', '#DCEDC2', '#DCEDC2',
            '#FFD3B5', '#FFD3B5', '#FFD3B5',
            '#FFAAA6', '#FFAAA6', '#FFAAA6',
            '#FF8C94',  '#FF8C94', '#FF8C94']
    fig, ax = plt.subplots(nrows=10, ncols=3)
    for i, ax in enumerate(ax.flatten()):
        df[df.columns[i]].plot(color=colors[i], ax=ax, sharex=ax)
        # ax.set_title(df.columns[i], fontsize=2)
        ax.tick_params(axis='both', which='major', labelsize=4)
        ax.tick_params(axis='both', which='minor', labelsize=2)
    plt.savefig('images/{}.png'.format(filename))
    plt.close()

### FUNCTIONS - Prepare Data
def do_prepare_data():
    main_df = pd.read_csv("tables/1_thirty_models.csv")
    model_keys=main_df['model_key'].unique().tolist()
    prepared_df = pd.DataFrame()
    for model_key in model_keys:
        df = load_data(main_df=main_df, model_key=model_key)
        df = fill_missing_days__and_set_datetime_index(df, start_date="2012-01-05", end_date="2016-12-28")
        df.to_csv("tables/_fill_days.csv")
        df.loc[:,'Order_Demand'] = mstats.winsorize(df['Order_Demand'].values, limits=[0.05, 0.05])
        # df['Order_Demand'] = df['Order_Demand'].apply(lambda x: np.log(x+1))
        df = moving_average_imputation(df)
        plot(df, model_key, '2_imputation_example')
        df['model_key'] = df['Order_Demand'].apply(lambda x: model_key)
        for i in range(1, 9):
            df['lag_{}'.format(i)] = df['Order_Demand'].shift(i)
        df = df.dropna()
        df['Date'] = df.index
        prepared_df = pd.concat([prepared_df, df])
    prepared_df = sqldf("select model_key, Date, Order_Demand, lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, lag_7, lag_8 from prepared_df ORDER BY 1,2", locals())
    prepared_df['Date'] = pd.to_datetime(prepared_df['Date'])
    prepared_df.to_csv("tables/2_data_prepared.csv", index=False)
    prepared_df.set_index(['Date'], inplace=True)
    thirty_plots(df=prepared_df, filename="3_prepared_data")

def load_data(main_df, model_key):
    return sqldf("SELECT Date, Order_Demand from main_df where model_key = '{}' ORDER BY 1".format(model_key), locals())

def fill_missing_days__and_set_datetime_index(df, start_date, end_date):
  idx = pd.date_range(start_date, end_date)
  df['Date'] = pd.to_datetime(df['Date'])
  df.set_index('Date', inplace=True)
  df = df.reindex(idx, fill_value=0)
  return df

def moving_average_imputation(df):
    # Consider +/- infinity here if working on a different time series
    df = df.replace(0, np.nan)
    df['Order_Demand'] = df['Order_Demand'].fillna(df['Order_Demand'].rolling(window=7, min_periods=1, center=False).mean())
    df['Order_Demand'] = df['Order_Demand'].fillna(method='ffill')
    return df

def plot(df, model_key, filename):
    fig, ax = plt.subplots()
    ax.set_title(model_key, fontsize=12)
    plt.plot(df['Order_Demand'])
    plt.savefig('images/{}.png'.format(filename))
    plt.close()

### FUNCTIONS - Spot Check Algorithms

def do_spot_check_algorithms():
    pass

def load_data_2(main_df, model_key):
    return sqldf("SELECT Date, Order_Demand, lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, lag_7, lag_8 from main_df where model_key = '{}' ORDER BY 1".format(model_key), locals())

def get_train_test_split(df):
    days_in_test_set = 10
    split_point = len(df) - days_in_test_set
    train, test = df[0:split_point], df[split_point:]
    return train, test

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # y_true, y_pred = np.expm1(y_true), np.expm1(y_pred)
    return np.mean(np.abs(y_true - y_pred) / np.abs(y_true))

##### MAIN

### ETL
# do_etl()

### Exploration
# do_exploration()

### Prepare data
# do_prepare_data()

### Spot Check Algorithms
main_df = pd.read_csv("tables/2_data_prepared.csv")
model_keys=main_df['model_key'].unique().tolist()
for model_key in model_keys:
    df = load_data_2(main_df=main_df, model_key=model_key)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['y'] = df['Order_Demand'].shift(-1)
    df = df.dropna()
    print(df.head())
    train, test = get_train_test_split(df)
    # Chose not to use CV because I don't want to tune the algorithms
    print("Persistence:")
    print(mean_absolute_percentage_error(test['y'], test['Order_Demand']))
    print("Average:")
    print(mean_absolute_percentage_error(test['y'], np.ones(10) * train['Order_Demand'].mean()))
    print("ML")


    X = df.drop(['y'], axis=1).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = df['y'].values

    models = []
    models.append(("LR",linear_model.LinearRegression()))
    models.append(("SGDRegressor",linear_model.SGDRegressor()))
    models.append(("ElasticNet",linear_model.ElasticNet()))
    models.append(("Ridge",linear_model.Ridge()))
    models.append(("KNN_1",neighbors.KNeighborsRegressor(n_neighbors=1)))
    models.append(("KNN_3",neighbors.KNeighborsRegressor(n_neighbors=3)))
    models.append(("KNN_5",neighbors.KNeighborsRegressor(n_neighbors=5)))
    models.append(("KNN_7",neighbors.KNeighborsRegressor(n_neighbors=7)))
    models.append(("KNN_11",neighbors.KNeighborsRegressor(n_neighbors=11)))
    models.append(("DT",tree.DecisionTreeRegressor()))
    models.append(("SVRLinear",svm.SVR(kernel='linear')))
    models.append(("SVRPoly",svm.SVR(kernel='poly')))
    models.append(("SVRRbf",svm.SVR(kernel='rbf')))
    models.append(("SVRSigmoid",svm.SVR(kernel='sigmoid')))
    bag_models = []
    for name, model in models:
        bag_models.append(("Bagging" + name,ensemble.BaggingRegressor(model,max_samples=0.5, max_features=0.5)))
    models = models + bag_models
    models.append(("ExtraTreesRegressor",ensemble.ExtraTreesRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=2)))
    models.append(("AdaBoost",ensemble.AdaBoostRegressor(n_estimators=50, random_state=2)))
    models.append(("GradientBoostingRegressor",ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)))
    models.append(("RandomForestRegressor",ensemble.RandomForestRegressor(n_estimators = 50, max_features="log2", min_samples_leaf=5, criterion="mse",bootstrap = True,random_state=2)))
    models.append(("XGBRegressor",make_pipeline(StackingEstimator(estimator=linear_model.RidgeCV()),
    XGBRegressor(learning_rate=0.1, max_depth=10, min_child_weight=13, n_estimators=40, nthread=1, subsample=0.55))))
    names, mses = [], []
    for name, model in models:
        cv_mse = cross_val_score(model, X, y, cv = KFold(n_splits=5, random_state=22), scoring='neg_mean_squared_error')
        names.append(name), mses.append(-1*cv_mse.mean())
    models_df = pd.DataFrame({'name': names, 'mse': mses}).sort_values(by=['mse']).iloc[0:]
    plt.close()
    ax = sns.barplot(x="name", y="mse", data=models_df)
    ax.set_xticklabels(models_df['name'], rotation=75, fontdict={'fontsize': 12})
    plt.savefig('images/models.png')
    plt.show()








# print(df.shape)
# print(df.head())
# print(df.dtypes)
# print(df.info())
# print(df.index)
