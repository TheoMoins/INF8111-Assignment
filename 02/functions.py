import re
import csv
import itertools
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def load_data(path, limit=None):
    with open(path, encoding='utf8') as file:
        reader = csv.reader(file, delimiter=',')
        # get header
        header = next(reader)
        data = [[value for value in row]
                for row in itertools.islice(reader, limit)]
    return np.asarray(header), np.asarray(data)


def print_sample(header, data, n=0):
    for i, (feature, value) in enumerate(zip(header, data[n])):
        print("({:^2d}) {:30} : {}".format(i, feature, value))


def print_feature(header, data, max_feature=5):
    for n_feature, feature in enumerate(data.T):
        values, counts = np.unique(feature, return_counts=True)
        counts_values = sorted(zip(counts, values), reverse=True)
        print("-" * 50)
        print("({:02d}) {} ({})".format(n_feature, header[n_feature],
                                        len(values)))
        print("-" * 50)
        for i, (v, c) in enumerate(counts_values):
            if i > max_feature:
                break
            print("{:10} : {:10} ({:5.1%})".format(c, v, v / data.shape[0]))


def delete_feature(header, data, feature_name):
    assert feature_name in header, "Index of {} does not exist".format(
        feature_name)

    index = np.where(header == feature_name)
    return np.delete(header, index), np.delete(data, index, 1)


def convert_date(header, data):
    assert "Date/Hour" in header, "Index of Date/Hour does not exist"

    new_data = []
    index = np.where(header == "Date/Hour")

    for i, d in enumerate(data):
        dt = datetime.fromisoformat(d[index][0])
        new_data.append(
            np.concatenate(
                (np.delete(d, index), [dt.year, dt.month, dt.day, dt.hour],
                 np.eye(7)[dt.date().weekday()])))

    date_header = [
        "Year", "Month", "Day", "Hour", "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]
    new_header = np.concatenate((np.delete(header, index), date_header))

    return np.asarray(new_header), np.asarray(new_data)


def convert_one_hot(header, data, feature_name):
    assert feature_name in header, "Index of {} does not exist".format(
        feature_name)

    index = np.where(header == feature_name)

    new_header, new_data = [], []

    mapping, enc = np.unique(data[:, index], return_inverse=True)

    add_header = [feature_name + " " + str(m) for m in mapping]

    new_header = np.concatenate((np.delete(header, index), add_header))

    for i, (d, e) in enumerate(zip(data, enc)):
        v = np.eye(mapping.shape[0])[e]
        new_data.append(np.concatenate((np.delete(d, index), v)))

    return np.asarray(new_header), np.asarray(new_data)


def convert_weather(header, data, weather):
    assert "Weather" in header, "Index of Weather does not exist"

    new_data = []
    N = len(weather)
    index = np.where(header == "Weather")

    for i, d in enumerate(data):
        new_weather = [
            1 if any([w == v for v in d[index][0].split(",")]) else 0
            for w in weather
        ]
        new_data.append(np.concatenate((np.delete(d, index), new_weather)))

    new_header = np.concatenate((np.delete(header, index), weather))

    return np.asarray(new_header), np.asarray(new_data)


def convert_type(data):
    return np.asarray(
        [[float(v.replace(",", ".")) for v in d] for d in data if "" not in d])


def normalization_feature(header, data, feature_name):
    assert feature_name in header, "Index of {} does not exist".format(
        feature_name)
    index = np.where(header == feature_name)
    data[:, index] = (data[:, index] - np.mean(data[:, index])) / np.std(
        data[:, index])


def split(header, data):
    y_index = np.where(header == "Withdrawals")
    l_index = np.where(header == "Volume")

    y = data[:, y_index].reshape(-1)
    label = data[:, l_index].reshape(-1)
    x = np.delete(data, (y_index, l_index), 1)

    header = np.delete(header, (y_index, l_index))

    return header, x, y, label


def plot_feature(header, x, feature_name):
    assert feature_name in header, "Index of {} does not exist".format(
        feature_name)
    index = np.where(header == feature_name)

    plt.figure(figsize=(6, 4), dpi=300)
    sns.distplot(x[:, index])
    plt.show()


def corr_matrix(header, x):
    mask = np.zeros((x.shape[1], x.shape[1]), dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    plt.figure(figsize=(14, 12), dpi=300)
    sns.heatmap(
        np.corrcoef(x.T),
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
    coeff = [np.corrcoef(feature, y)[0][1] for feature in x.T]
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