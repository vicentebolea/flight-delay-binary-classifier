import pandas as pd


def get_season(month):
    if month >= 12 or month <= 2:
        return 0
    elif month <= 5:
        return 1
    elif month <= 8:
        return 2
    else:
        return 3


def is_delay(row):
    if row['Cancelled'] == 1 or row['ArrTime'] > 15:
        return 1
    else:
        return 0


df2008 = pd.read_csv("dataset/2008.csv")
df2007 = pd.read_csv("dataset/2007.csv")
df2006 = pd.read_csv("dataset/2006.csv")

df = pd.concat([df2008, df2007, df2006])
df['ArrDelay'] = df.apply(is_delay, axis=1)
columns = ['Year', 'DepTime', 'ArrTime', 'TailNum', 'ActualElapsedTime', 'AirTime',
           'DepDelay', 'Distance', 'TaxiIn', 'TaxiOut', 'Cancelled', 'CancellationCode',
           'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay',
           'LateAircraftDelay']
df.drop(columns, inplace=True, axis=1)
df.dropna(inplace=True)
df['Season'] = df['Month'].apply(lambda x: get_season(x))
df['Month'] = df['Month'].apply(lambda x: x - 1)
df['DayofMonth'] = df['DayofMonth'].apply(lambda x: x - 1)
df['DayOfWeek'] = df['DayOfWeek'].apply(lambda x: x - 1)
train = df.sample(frac=0.85, random_state=200)
test = df.drop(train.index)
test.to_csv('test.csv', sep=',', index=False)
train.sort_index(inplace=True)
train.to_csv('train.csv', sep=',', index=False)
