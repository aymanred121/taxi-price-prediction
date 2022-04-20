from functions import *


weatheDf = pd.read_csv("taxi/weather.csv")
lst_location = weatheDf.location.unique()
weatheDf.location = weatheDf.location.map(weatheDf.groupby('location')['rain'].mean())
weatheDf.time_stamp = pd.to_datetime(weatheDf.time_stamp,unit='s')
dict_loc = dict(zip(weatheDf.location.unique(),lst_location))

weatheDf['month'] = weatheDf.time_stamp.dt.month
weatheDf.time_stamp = weatheDf.time_stamp.dt.hour
weatheDf =  weatheDf.rename(
    columns ={
        'time_stamp':'hour'
    }
    )

scaler = MinMaxScaler()
scaler.fit(weatheDf.drop('rain',axis=1))
wweatheDf = pd.DataFrame(scaler.transform(weatheDf.drop('rain',axis=1)), columns= weatheDf.drop('rain',axis=1).columns)
weatheDf1 =weatheDf.copy()
weatheDf.dropna(axis=0,inplace=True)
X = weatheDf.drop('rain',axis=1)
y = weatheDf.rain
x_pred = weatheDf1.drop('rain',axis=1)

model = linear_model.LinearRegression()
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X)
x_pred_poly = poly_features.fit_transform(x_pred)
model.fit(X_train_poly, y)
y_train_predicted = model.predict(x_pred_poly)

weatheDf1.rain = y_train_predicted

weatherData =weatheDf1
weatherData.location= weatherData.location.replace(dict_loc)
taxiRidersData = pd.read_csv("taxi/taxi-rides.csv")
taxiRidersData.time_stamp = pd.to_datetime(taxiRidersData.time_stamp,unit="ms")
taxiRidersData['month']=(taxiRidersData.time_stamp.dt.month)
taxiRidersData.time_stamp = taxiRidersData.time_stamp.dt.hour
taxiRidersData.rename(
    columns ={
        'time_stamp':'hour'
    }
)

taxiRidersData.dropna(axis=0, inplace=True)
avgWeather = weatherData.groupby(["location"]).mean().reset_index(drop=False)
print(avgWeather.columns)
avgWeather.drop(['hour'],axis=1,inplace=True)
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
print(to_drop)
data = data.drop(to_drop, axis=1)

dataTrain,dataTest = split_stratified_into_train_val_test(data,stratify_colname='price', frac_train=0.60, frac_test=0.20)

dataTrain = cap_data(dataTrain)
dataTest = cap_data(dataTest)

dataTrain = mean_encoding(dataTrain)
dataTest = mean_encoding(dataTest)

corr = dataTest.corr()
plt.figure(figsize=(16, 5))
dataplot = sns.heatmap(corr, annot=True, linewidths=1)
top_feature = corr.index[abs(corr['price']) > 0.2]
print(top_feature)
dataTest = dataTest[top_feature]
dataTrain = dataTrain[top_feature]

X_train = dataTrain.drop(['price'],axis=1)
X_test = dataTest.drop(['price'],axis=1)
y_train = dataTrain.price
y_test = dataTest.price

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = pd.DataFrame(scaler.transform(X_train), columns= dataTrain.drop(['price'],axis=1).columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns= dataTrain.drop(['price'],axis=1).columns)

deg = 8
model_trial(X_train, X_test, y_train, y_test, linear_model.LinearRegression(),deg)
model_trial(X_train, X_test, y_train, y_test, linear_model.Ridge(), deg)
corssValidation(X_train, X_test, y_train, y_test, linear_model.LinearRegression(),deg)
corssValidation(X_train, X_test, y_train, y_test,linear_model.Ridge(),deg)

