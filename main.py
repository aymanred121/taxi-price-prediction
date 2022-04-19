import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import *
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import *
from  scipy import stats
from sklearn.model_selection import cross_val_score

def Remove_Outlier_Indices(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    trueList = ~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR)))
    return trueList


def corssValidation(X_train, X_test, y_train, y_test,model, degree=1):
    modelfeatures = PolynomialFeatures(degree)
    # transforms the existing features to higher degree features.
    X_train_poly_model_1 = modelfeatures.fit_transform(X_train)
    # fit the transformed features to Linear Regression
    poly_model1 = model
    scores = cross_val_score(poly_model1, X_train_poly_model_1, y_train, scoring='neg_mean_squared_error', cv=9)
    model_1_score = abs(scores.mean())
    poly_model1.fit(X_train_poly_model_1, y_train)
    print("model cross validation score is "+ str(model_1_score))
    # predicting on test data-set
    prediction = poly_model1.predict(modelfeatures.fit_transform(X_test))
    print('model Test Mean Square Error', metrics.mean_squared_error(y_test, prediction))


def model_trial(X_train, X_test, y_train, y_test, model, degree=30):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)

    model.fit(X_train_poly, y_train)

    y_train_predicted = model.predict(X_train_poly)
    prediction = model.predict(poly_features.fit_transform(X_test))
    train_err = metrics.mean_squared_error(y_train, y_train_predicted)
    test_err = metrics.mean_squared_error(y_test, prediction)
    print('Train subset (MSE) for degree {}: '.format(degree), train_err)
    print('Test subset (MSE) for degree {}: '.format(degree), test_err)

weatherData = pd.read_csv("taxi/weather.csv")
taxiRidersData = pd.read_csv("taxi/taxi-rides.csv")
taxiRidersData.time_stamp = pd.to_datetime(taxiRidersData.time_stamp,unit="ms")
taxiRidersData.time_stamp = taxiRidersData.time_stamp.dt.hour
weatherData.time_stamp = pd.to_datetime(weatherData.time_stamp,unit='s')
weatherData.time_stamp = weatherData.time_stamp.dt.hour
weatherData.dropna()
taxiRidersData.dropna(axis=0, inplace=True)
avgWeather = weatherData.groupby("location").mean().reset_index(drop=False)
avgWeather.drop(['time_stamp'],axis=1,inplace=True)
sourceWeather = avgWeather.rename(
    columns={
        'location': 'source',
        'rain': 'source_rain',
        'temp': 'source_temp',
        'clouds': 'source_clouds',
        'pressure': 'source_pressure',
        'humidity': 'source_humidity',
        'wind': 'source_wind'
    }
)

destinationWeather = avgWeather.rename(
    columns={
        'location': 'destination',
        'rain': 'destination_rain',
        'temp': 'destination_temp',
        'clouds': 'destination_clouds',
        'pressure': 'destination_pressure',
        'humidity': 'destination_humidity',
        'wind': 'destination_wind'

    }
)
data = taxiRidersData.merge(sourceWeather, on='source')\
    .merge(destinationWeather, on="destination")
for i  in ((data.product_id + " " + data.name).unique()):
    print(i)
data.drop(['product_id'],axis=1,inplace=True)
data.drop(['id'],axis=1,inplace=True)

corr = data.corr()
plt.figure(figsize=(16, 5))
dataplot = sns.heatmap(data.corr(), annot=True, linewidths=1)

dataX = data.drop(['price'],axis=1)
cor_matrix = dataX.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.92)]
data = data.drop(to_drop, axis=1)

cols = ['distance',
'source_clouds',
'source_temp',
'source_pressure',
'source_rain',
'destination_clouds',
'destination_temp',
'destination_pressure',
'destination_rain'
]
data = data[Remove_Outlier_Indices(data).all(1)]

data.name = data.name.map(data.groupby('name')['price'].mean())
data.source = data.source.map(data.groupby('source')['price'].mean())
data.destination = data.destination.map(data.groupby('destination')['price'].mean())
data.cab_type = data.cab_type.map(data.groupby('cab_type')['price'].mean())

X = data.drop(['price'], axis=1)
Y = data['price']


corr = data.corr()
plt.figure(figsize=(16, 5))
dataplot = sns.heatmap(data.corr(), annot=True, linewidths=1)
top_feature = corr.index[abs(corr['price']) > 0.1]
top_feature = top_feature.drop("price")
print(top_feature)
X = X[top_feature]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)


model_trial(X_train, X_test, y_train, y_test, linear_model.LinearRegression(),degree=12)
model_trial(X_train, X_test, y_train, y_test, linear_model.Ridge(), degree=12)



corssValidation(X_train, X_test, y_train, y_test, linear_model.LinearRegression(),degree=12)
corssValidation(X_train, X_test, y_train, y_test,linear_model.Ridge(),degree=12)