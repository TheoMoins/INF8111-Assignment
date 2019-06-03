import gc
import re
import csv
import time
import itertools
import statistics
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split


def load_data(path, limit=None):
    with open(path, encoding='utf8') as file:
        reader = csv.reader(file, delimiter=',')
        # get header
        header = next(reader)
        data = [[value for value in row]
                for row in itertools.islice(reader, limit)]
    return header, data


def print_sample(header, data, n=0):
    for i, (feature, value) in enumerate(zip(header, data[n])):
        print("({:^2d}) {:30} : {}".format(i, feature, value))


def print_feature(header, data, max_feature=5):
    for n_feature, feature in enumerate(zip(*data)):
        values, counts = np.unique(feature, return_counts=True)
        counts_values = sorted(zip(counts, values), reverse=True)
        print("-" * 50)
        print("({:02d}) {} ({})".format(n_feature, header[n_feature],
                                        len(values)))
        print("-" * 50)
        for i, (v, c) in enumerate(counts_values):
            if i > max_feature:
                break
            print("{:10} : {:10} ({:5.1%})".format(c, v, v / len(data)))


def delete_feature(header, data, feature_name):
    index = header.index(feature_name)
    del header[index]
    data = list(zip(*data))
    del data[index]
    data = list(map(list, zip(*data)))
    return header, data


def convert_date(header, data):
    index = header.index("Date/Hour")
    header += ["Year", "Month", "Day", "Hour", "Weekday"]
    for d in data:
        dt = datetime.fromisoformat(d[index])
        d += [dt.year, dt.month, dt.day, dt.hour, dt.date().weekday()]

    return delete_feature(header, data, "Date/Hour")


def convert_one_hot(header, data, feature_name):
    index = header.index(feature_name)
    values = list(set(list(zip(*data))[index]))
    header += [feature_name + " " + str(v) for v in values]
    for d in data:
        d += [1 if v == d[index] else 0 for v in values]

    return delete_feature(header, data, feature_name)


def convert_weather(header, data, weather):
    N = len(weather)
    index = header.index("Weather")
    header += weather
    for d in data:
        d += [
            1 if any([w == v for v in d[index].split(",")]) else 0
            for w in weather
        ]

    return delete_feature(header, data, "Weather")

def convert_weather_coef(header, data, weather, weather_coef):
    header+= ["Weather_Coef"]
    ind_col = [header.index(w) for w in weather]
    for d in data:
        d_weather = d[ind_col[0]:ind_col[-1]]
        if sum(d_weather)!= 0:
            wc_d = sum([a*b for (a,b) in zip (d_weather, weather_coef)])/sum(d_weather)
            d += [wc_d]
        else:
            d += [-1]
    for w in weather:
        header, data = delete_feature(header, data, w)
    return header, data


def remove_missing(data):
    return [d for d in data if "" not in d]


def convert_type(data):
    return [[
        float(v.replace(",", ".")) if isinstance(v, str) else v for v in d
    ] for d in data]


def normalization_feature(header, data, feature_name):
    index = header.index(feature_name)
    data = list(map(list, zip(*data)))
    mean = statistics.mean(data[index])
    std = statistics.stdev(data[index])
    data = list(map(list, zip(*data)))
    for d in data:
        d[index] = (d[index] - mean) / std

    return header, data


def split(header, data):
    y_index = header.index("Withdrawals")
    l_index = header.index("Volume")
    data = list(map(list, zip(*data)))
    y = data[y_index]
    label = data[l_index]
    data = list(map(list, zip(*data)))
    header, data = delete_feature(header, data, "Withdrawals")
    header, x = delete_feature(header, data, "Volume")

    return header, x, y, label


def plot_feature(header, x, feature_name):
    index = header.index(feature_name)
    plt.figure(figsize=(6, 4), dpi=300)
    sns.distplot(list(map(list, zip(*x)))[index])
    plt.show()


def corr_matrix(header, x):
    mask = np.zeros((len(x[0]), len(x[0])), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=(14, 12), dpi=300)
    sns.heatmap(
        np.corrcoef(list(map(list, zip(*x)))),
        mask=mask,
        center=0,
        cmap=cmap,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        xticklabels=header,
        yticklabels=header)
    plt.show()


def feature_output_corr(header, x, y, limit=None):
    coeff = [
        np.corrcoef(feature, y)[0][1] for feature in list(map(list, zip(*x)))
    ]
    abs_coeff = list(map(abs, coeff))
    for _, coeff, name in itertools.islice(
            sorted(zip(abs_coeff, coeff, header), reverse=True), limit):
        print("{:30} : {:6.3f}".format(name, coeff))


def compute_f1(proba, y_true, step=0.01, plot=False):
    f1 = []

    for threshold in np.arange(0, 1, step):
        y_pred = [int(y > threshold) for y in proba]
        f1.append(f1_score(y_true, y_pred) if 1 in y_pred else 0)

    if plot:
        plt.figure(figsize=(6, 4), dpi=300)
        plt.plot(np.arange(0, 1, step), f1)
        plt.xlabel('Threshold')
        plt.ylabel('F1-score')
        plt.show()

    return max(f1), step * np.argmax(f1)


def missing_to_value(header, data, feature_name, new_value):
    index = header.index(feature_name)
    for d in data:
        if d[index] == "":
            d[index] = new_value
    return header, data

def sort_by_duration(header, x, y=None, label=None):
    index_y = header.index("Year")
    index_m = header.index("Month")
    index_d = header.index("Day")
    index_h = header.index("Hour")
    
    time = [v[index_h]+v[index_d]*24+v[index_m]*24*31+v[index_y]*24*365 for v in x]
    
    x= list(zip(*sorted(list(zip(time, x)))))[1]
    if y and label:
        y = list(zip(*sorted(list(zip(time, y)))))[1]
        label = list(zip(*sorted(list(zip(time, label)))))[1]
    
    return x, y, label

def sort_by_station(header, x, y=None, label=None):
    index = header.index("Station Code")
    stations = list(set(list(zip(*x))[index]))
    x_stations = [[] for _ in stations]

    if y and label:
        y_stations = [[] for _ in stations]
        label_stations = [[] for _ in stations]
        for _x, _y, _label in zip(x, y, label):
            s = stations.index(_x[index])
            x_stations[s].append(_x)
            y_stations[s].append(_y)
            label_stations[s].append(_label)
        return header, stations, x_stations, y_stations, label_stations
    else:
        for _x in x:
            s = stations.index(_x[index])
            x_stations[s].append(_x)
        return header, stations, x_stations


def pipeline(path="data/training.csv",
             limit=None,
             delete_features=["Visility indicator", "hmdx", "Wind Chill"],
             cvrt_date=True,
             weather=[
                 "Orages", "Brouillard", "Bruine", "Généralement dégagé",
                 "Généralement nuageux", "Pluie", "Pluie modérée",
                 "Pluie forte", "Dégagé", "Nuageux", "Neige"
             ],
             one_hot_features=["Weekday"],
             norm_features=[
                 "Temperature (°C)", "Drew point (°C)",
                 "Relativite humidity (%)", "wind direction (10s deg)",
                 "Wind speed (km/h)", "Pressure at the station (kPa)",
                 "Visibility (km)"
             ],
             missing_features=['wind direction (10s deg)'],
             missing_values=[23],
             test=False,
             weather_coef = []):
    """
    path :           (STRING) path of the file to load.
    limit:           (INT) limit the number of example to load.
    delete_features: (LIST) feature names to remove.
    cvrt_date:       (BOOLEAN) convert the data
    weather:         (LIST) weather to consider. All other will be dropped.
    one_hot_features (LIST) feature names to convert in one-hot vector.
    norm_features    (LIST) feature names to normalize in one-hot vector
    missing_features (LIST) feature which missing values are to replace 
    missing_values   (LIST) value with which to replace the missing values
    """
    start = time.time()
    header, data = load_data(path, limit)
    print("Data loaded ({:.1f}s)".format(time.time() - start))

    for f in delete_features:
        start = time.time()
        header, data = delete_feature(header, data, f)
        print("{} deleted ({:.1f}s)".format(f, time.time() - start))

    if cvrt_date:
        start = time.time()
        header, data = convert_date(header, data)
        print("Date splited in Year/Month/Day/Hour/Weekday ({:.1f}s)".format(
            time.time() - start))

    for f in one_hot_features:
        start = time.time()
        header, data = convert_one_hot(header, data, f)
        print("{} converted in one-hot vector ({:.1f}s)".format(
            f,
            time.time() - start))

    if weather:
        start = time.time()
        header, data = convert_weather(header, data, weather)
        print("Weather converted ({:.1f}s)".format(time.time() - start))
       
    if weather_coef:
        start = time.time()
        header, data = convert_weather_coef(header, data, weather, weather_coef)
        print("Weather rescaled ({:.1f}s)".format(time.time() - start))

    for f, v in zip(missing_features, missing_values):
        start = time.time()
        header, data = missing_to_value(header, data, f, v)
    print("Replace missing values ({:.1f}s)".format(time.time() - start))

    start = time.time()
    data = remove_missing(data)
    print("Remove samples with missing values ({:.1f}s)".format(time.time() -
                                                                start))
    start = time.time()
    data = convert_type(data)
    print("Data converted to float ({:.1f}s)".format(time.time() - start))

    for f in norm_features:
        start = time.time()
        header, data = normalization_feature(header, data, f)
        print("{} normalized ({:.1f}s)".format(f, time.time() - start))

    start = time.time()
    index = header.index('Station Code')
    data.sort(key=lambda x: x[index])
    print("Sort data according to station code ({:.1f}s)".format(time.time() -
                                                                 start))

    if not test:
        start = time.time()
        header, x, y, label = split(header, data)
        print("split data into x, y, and label ({:.1f}s)".format(time.time() -
                                                             start))
        gc.collect()
        return header, x, y, label

    else:
        return header, data