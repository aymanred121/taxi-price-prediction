# %%
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
from scipy import stats
from sklearn.model_selection import cross_val_score
from category_encoders import TargetEncoder
import math
import pickle


# %% [markdown]
# ## Weather Data PRE-PROCESSING

# %%
encoder = TargetEncoder()
weatheDf = pd.read_csv("weather.csv")
weatheDf["encodedlocation"] = encoder.fit_transform(
    weatheDf['location'], weatheDf['rain'])
weatheDf.time_stamp = pd.to_datetime(weatheDf.time_stamp, unit='s')


# %% [markdown]
# #### remove time_stamp and replace it with hour and month columns instead

# %%
#weatheDf.month = weatheDf.time_stamp.dt.month
weatheDf['month'] = weatheDf.time_stamp.dt.month
weatheDf.time_stamp = weatheDf.time_stamp.dt.hour
weatheDf = weatheDf.rename(
    columns={
        'time_stamp': 'hour'
    }
)

# %%
weatheDf.columns

# %% [markdown]
# #### Scale the data using minMaxScaler

# %%
scaler = MinMaxScaler()
scaler.fit(weatheDf.drop(['rain', 'location'], axis=1))
#wweatheDf = pd.DataFrame(scaler.transform(weatheDf.drop('rain',axis=1)), columns= weatheDf.drop('rain',axis=1).columns)
weatheDf1 = weatheDf.copy()
weatheDf.dropna(axis=0, inplace=True)
X = weatheDf.drop(['rain', 'location'], axis=1)
y = weatheDf.rain
x_pred = weatheDf1.drop(['rain', 'location'], axis=1)

# %% [markdown]
# #### Train the model to predict the missing values of rain feature

# %%
model = linear_model.LinearRegression()
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X)
x_pred_poly = poly_features.fit_transform(x_pred)
model.fit(X_train_poly, y)
y_train_predicted = model.predict(x_pred_poly)


# %%
y_train_predicted.shape
y_train_predicted
weatheDf1.rain = y_train_predicted


# %%
def split_stratified_into_train_val_test(df_input, stratify_colname='y',
                                         frac_train=0.8, frac_test=0.2,
                                         random_state=None):
    X = df_input  # Contains all columns.
    # Dataframe of just the column on which to stratify.
    y = df_input[[stratify_colname]]

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          test_size=(
                                                              1.0 - frac_train),
                                                          random_state=random_state)

    assert len(df_input) == len(df_train) + len(df_temp)

    return df_train, df_temp

# %%


def Feature_Encoder(X, cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


# %%
def one_hot(df, cols):
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    return df


# %% [markdown]
# ## working on the Main Dataset

# %%
def test_sample(sampleData):
    sampleData.drop(['price'], axis=1, inplace=True)
    sampleData.drop(['id'], axis=1, inplace=True)
    y_tru = sampleData['price']
    sampleData = sampleData[top_feature]
    sampleData.drop(['price'], axis=1, inplace=True)
    for col in sampleData.select_dtypes(include='O').columns:
        encoder = pickle.load(open(col + '.pkl', 'rb'))
        sampleData[col] = encoder.transform(sampleData[col])
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    scaler.fit(sampleData)
    sampleData = pd.DataFrame(scaler.transform(sampleData))
    clrf = pickle.load(open('linear.pkl', 'rb'))
    sample_pred = clrf.predict(sampleData)
    print('linear acc', metrics.accuracy_score(y_tru, sample_pred) * 100.0)


# %%
weatherData = weatheDf1.drop('encodedlocation', axis=1)
taxiRidersData = pd.read_csv("taxi-rides.csv")
taxiRidersData.time_stamp = pd.to_datetime(
    taxiRidersData.time_stamp, unit="ms")
#taxiRidersData.month =taxiRidersData.time_stamp.dt.month
taxiRidersData['month'] = (taxiRidersData.time_stamp.dt.month)
taxiRidersData.time_stamp = taxiRidersData.time_stamp.dt.hour
taxiRidersData.rename(
    columns={
        'time_stamp': 'hour'
    }
)


# %%
taxiRidersData.dropna(axis=0, inplace=True)
avgWeather = weatherData.groupby(["location"]).mean().reset_index(drop=False)
print(taxiRidersData.columns)
avgWeather.drop(['hour'], axis=1, inplace=True)
avgWeather.drop(['month'], axis=1, inplace=True)
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


# %%
data = taxiRidersData.merge(sourceWeather, on='source')\
    .merge(destinationWeather, on="destination")
print(data.columns)

# %% [markdown]
# ### DATA PRE-PROCESSING
#

# %% [markdown]
# since product_id and name have 100% correlation we can remove one of them as it is considered dublication

# %%
# for i  in ((data.product_id + " " + data.name).unique()):
#     #print(i)
data.drop(['product_id'], axis=1, inplace=True)
data.drop(['id'], axis=1, inplace=True)
print(data.columns)

# %% [markdown]
# from the figer we can see that thier are multipl features that are considered highly correlated with each other that we can drop which are:
#
# * source_humidity -> source_temp
# * source_wind ->  source_cloud
# * destination_humidity -> destination_temp
# * destination_wind -> destination_cloud

# %%
corr = data.corr()
plt.figure(figsize=(16, 5))
dataplot = sns.heatmap(data.corr(), annot=True, linewidths=1)

# %%
dataX = data.drop(['price'], axis=1)
cor_matrix = dataX.corr().abs()
upper_tri = cor_matrix.where(
    np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(
    upper_tri[column] > 0.92)]
print(to_drop)
data = data.drop(to_drop, axis=1)

# %% [markdown]
# ### remove outliers

# %%


def cap_data(df):
    for col in df.columns:
        if (((df[col].dtype) == 'float64') | ((df[col].dtype) == 'int64')):
            percentiles = df[col].quantile([0.25, 0.75]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col] = df[col]
    return df

# %% [markdown]
# ### Encdoing Data


# %%
dataTrain, dataTest = split_stratified_into_train_val_test(
    data, stratify_colname='price', frac_train=0.60, frac_test=0.20)

# %%
dataTrain = cap_data(dataTrain)
dataTest = cap_data(dataTest)

# %% [markdown]
# Using mean Encdoing to encode data since it has great balance between efficency and model complexity

# %%


def mean_encoding(data):
    for col in data.select_dtypes(include='O').columns:
        encoder = TargetEncoder()
        data[col] = encoder.fit_transform(data[col], data['price'])
        pickle.dump(encoder, open(f'{col}.pkl', 'wb'))
    return data


# %%
dataTrain = mean_encoding(dataTrain)
dataTest = mean_encoding(dataTest)

# %%
corr = dataTest.corr()
plt.figure(figsize=(16, 5))
dataplot = sns.heatmap(corr, annot=True, linewidths=1)
top_feature = corr.index[abs(corr['price']) > 0.2]
print(top_feature)
dataTest = dataTest[top_feature]
dataTrain = dataTrain[top_feature]


# %% [markdown]
# ### Model training and result

# %%
#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)
X_train = dataTrain.drop(['price'], axis=1)
X_test = dataTest.drop(['price'], axis=1)
y_train = dataTrain.price
y_test = dataTest.price

# %%
scaler = MinMaxScaler()
scaler.fit(X_train)
pickle.dump(scaler, open('scaler.pkl', 'wb'))
X_train = pd.DataFrame(scaler.transform(
    X_train), columns=dataTrain.drop(['price'], axis=1).columns)
X_test = pd.DataFrame(scaler.transform(
    X_test), columns=dataTest.drop(['price'], axis=1).columns)


# %%
def model_trial(X_train, X_test, y_train, y_test, model, degree=30):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)

    model.fit(X_train_poly, y_train)
    pickle.dump(model, open('linear.pkl', 'wb'))
    y_train_predicted = model.predict(X_train_poly)
    prediction = model.predict(poly_features.fit_transform(X_test))
    train_err = metrics.mean_squared_error(y_train, y_train_predicted)
    test_err = metrics.mean_squared_error(y_test, prediction)
    train_acc = metrics.r2_score(y_train, y_train_predicted)
    test_acc = metrics.r2_score(y_test, prediction)
    print('Train subset (MSE) for degree {}: '.format(degree), train_err)
    print('Test subset (MSE) for degree {}: '.format(degree), test_err)
    # print("R^2 for train ", train_acc)
    # print("R^2 for test ", test_acc)


# %%
def corssValidation(X_train, X_test, y_train, y_test, model, degree=1):
    modelfeatures = PolynomialFeatures(degree)
    # transforms the existing features to higher degree features.
    X_train_poly_model_1 = modelfeatures.fit_transform(X_train)
    # fit the transformed features to Linear Regression
    poly_model1 = model
    scores = cross_val_score(poly_model1, X_train_poly_model_1,
                             y_train, scoring='neg_mean_squared_error', cv=9)
    model_1_score = abs(scores.mean())
    poly_model1.fit(X_train_poly_model_1, y_train)
    pickle.dump(poly_model1, open('cross.pkl', 'wb'))
    print("model cross validation score is " + str(model_1_score))
    # predicting on test data-set
    prediction = poly_model1.predict(modelfeatures.fit_transform(X_test))
    print('model Test Mean Square Error',
          metrics.mean_squared_error(y_test, prediction))


# %%
deg = 9
model_trial(X_train, X_test, y_train, y_test,
            linear_model.LinearRegression(), deg)


# %%
test_sample(pd.read_csv('taxi-test-samples.csv'))

# %%
#model_trial(X_train, X_test, y_train, y_test, linear_model.Ridge(), deg)


# %%
corssValidation(X_train, X_test, y_train, y_test,
                linear_model.LinearRegression(), degree=5)
corssValidation(X_train, X_test, y_train, y_test,
                linear_model.Ridge(), degree=5)
