from ucimlrepo import fetch_ucirepo
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from prettytable import PrettyTable
import numpy as np
import time



# Импорт датасета
wholesale_customers = fetch_ucirepo(id=292)
x = wholesale_customers.data.features
y = wholesale_customers.data.targets



# Нормализация датасета
x_train = np.array(x)
x_scaled = preprocessing.StandardScaler().fit_transform(x_train)
y = np.array(y).ravel()



# Заполнение таблицы собственных векторов и значений
pca = PCA()
X = pca.fit_transform(x_scaled)
eigenvectors = pca.fit_transform(X)
eigenvalues = pca.explained_variance_



# Создание "таблиц" для представления результатов
table_knc  = PrettyTable()
table_rnc  = PrettyTable()
table_nc   = PrettyTable()
table_dtc  = PrettyTable()
table_rfc = PrettyTable()
table_gnb = PrettyTable()
table_svc = PrettyTable()
table_best = PrettyTable()

# Добавление колонок в "таблицы"
table_knc.field_names  = ["Размерность пространства", "Точность", "Время выполнения", "k",               "Вес"        ]
table_rnc.field_names  = ["Размерность пространства", "Точность", "Время выполнения", "Радиус",          "Вес"        ]
table_nc.field_names   = ["Размерность пространства", "Точность", "Время выполнения", "Метрика",         "-"          ]
table_dtc.field_names  = ["Размерность пространства", "Точность", "Время выполнения", "Критерий",        "Разделитель"]
table_rfc.field_names  = ["Размерность пространства", "Точность", "Время выполнения", "Кол-во деревьев", "Критерий"   ]
table_gnb.field_names  = ["Размерность пространства", "Точность", "Время выполнения", "Сглаживание",     "-"          ]
table_svc.field_names  = ["Размерность пространства", "Точность", "Время выполнения", "Ядро",            "Степень"    ]
table_best.field_names = ["Размерность пространства", "Точность", "Время выполнения", "Параметр 1",      "Параметр 2" ]

# Создание массивов для поиска лучших результатов
best_knc = [0, 0, 0, "", ""]
best_rnc = [0, 0, 0, "", ""]
best_nc  = [0, 0, 0, "", ""]
best_dtc = [0, 0, 0, "", ""]
best_rfc = [0, 0, 0, "", ""]
best_gnb = [0, 0, 0, "", ""]
best_svc = [0, 0, 0, "", ""]



def search_for_best_results (table, result, dimension, accuracy, period, parameter_1, parameter_2):
    if (result[1] < accuracy) or (result[1] == accuracy and result[2] > period):
        result[0] = dimension
        result[1] = accuracy
        result[2] = period
        result[3] = parameter_1
        result[4] = parameter_2

    table.add_row([dimension, accuracy, period, parameter_1, parameter_2])



def table_best_formation(table, best):
    table.add_row([best[0], best[1], best[2], best[3], best[4]])



# Цикл по всем размерностям
for component in range(1, len(eigenvalues) + 1):
    pca = PCA(n_components=component)
    X = pca.fit_transform(x_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



    # Метод k-ближайших соседей
    for weight in ["uniform", "distance"]:
        for k in range(1, 30, 2):
            knc = KNeighborsClassifier(n_neighbors=k, weights=weight)

            start_time = time.time()
            scores = cross_val_score(knc, X_train, y_train, cv=10)
            end_time = time.time()

            search_for_best_results(table_knc, best_knc, component, scores.mean(), end_time - start_time, k, weight)



    # Метод радиус-ближайших соседей
    for weight in ["uniform", "distance"]:
        for radius in range(15, 25):
            rnc = RadiusNeighborsClassifier(radius=radius, weights=weight)

            start_time = time.time()
            scores = cross_val_score(rnc, X_train, y_train, cv=10)
            end_time = time.time()

            search_for_best_results(table_rnc, best_rnc, component, scores.mean(), end_time - start_time, radius, weight)



    # Метод центройда
    for metric in ["euclidean", "manhattan"]:
        nc = NearestCentroid(metric=metric)

        start_time = time.time()
        scores = cross_val_score(nc, X_train, y_train, cv=10)
        end_time = time.time()

        search_for_best_results(table_nc, best_nc, component, scores.mean(), end_time - start_time, metric, "-")



    # Метод дерева решений
    for criterion in ["gini", "entropy", "log_loss"]:
        for splitter in ["best", "random"]:
            dtc = DecisionTreeClassifier(criterion=criterion, splitter=splitter)

            start_time = time.time()
            scores = cross_val_score(dtc, X_train, y_train, cv=10)
            end_time = time.time()

            search_for_best_results(table_dtc, best_dtc, component, scores.mean(), end_time - start_time, criterion, splitter)



    # Метод случайного леса
    for n in range(20, 200, 20):
        for criterion in ["gini", "entropy", "log_loss"]:
            rfc = RandomForestClassifier(n_estimators=n, criterion=criterion)

            start_time = time.time()
            scores = cross_val_score(rfc, X_train, y_train, cv=10)
            end_time = time.time()

            search_for_best_results(table_rfc, best_rfc, component, scores.mean(), end_time - start_time, n, criterion)



    # Метод наивного Байеса
    for var_smoothing in range(-10, 11):
        gnb = GaussianNB(var_smoothing=(pow(10, var_smoothing)))

        start_time = time.time()
        scores = cross_val_score(gnb, X_train, y_train, cv=10)
        end_time = time.time()

        search_for_best_results(table_gnb, best_gnb, component,scores.mean(), end_time - start_time, var_smoothing, "-")



    # Метод машины опорных векторов
    for kernel in ["linear", "poly", "rbf", "sigmoid"]:
        for n in range(5):
            svc = SVC(kernel=kernel, degree=n)

            start_time = time.time()
            scores = cross_val_score(svc, X_train, y_train, cv=10)
            end_time = time.time()

            search_for_best_results(table_svc, best_svc, component, scores.mean(), end_time - start_time, kernel, n)



table_best_formation(table_best, best_knc)
table_best_formation(table_best, best_rnc)
table_best_formation(table_best,  best_nc)
table_best_formation(table_best, best_dtc)
table_best_formation(table_best, best_rfc)
table_best_formation(table_best, best_gnb)
table_best_formation(table_best, best_svc)



print("Метод k-ближайших соседей")
print(table_knc)

print("\nМетод радиус-ближайших соседей")
print(table_rnc)

print("\nМетод центройда")
print(table_nc)

print("\nМетод дерева решений")
print(table_dtc)

print("\nМетод случайного леса")
print(table_rfc)

print("\nМетод наивного Байеса")
print(table_gnb)

print("\nМетод машины опорных векторов")
print(table_svc)

print("\nЛучшие результаты")
print(table_best)