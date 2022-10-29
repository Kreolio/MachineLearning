import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from time import time


# Загрузим датасет цифр
digits = datasets.load_digits()

# Вычленим матрицу (1797, 64), где каждая строка имеет 64 признака
# А также вычленим соответствующие этим 1797 строчкам цифры (объекты)
X = digits.data
Y = digits.target
# Количество уникальных элементов (10 цифр)
unique = len(set(Y))

print("Размерность данных: ", np.shape(X))
print("Количество признаков: ", np.shape(X)[1])
print("Количество объектов: ", np.shape(X)[0])
print("Количество уникальных значений (цифр): ", unique)
print("\n")

# Функция, считающие метрики 
def k_alg(kmeans, name, data, labels):

    t0 = time()
    # Объединим два метода в трубку для применения к data
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    t1 = time()
    fit_time = t1 - t0

    metrics_of_alg = [
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score
    ]
    values_of_metrics = [m(labels, estimator[-1].labels_) for m in metrics_of_alg]

    print("Случай init = {}\n".format(name))
    print("Метрика ARI: ", values_of_metrics[0])
    print("Метрика AMI: ", values_of_metrics[1])
    print("Время алгоритма: ", fit_time)
    print("\n")

# Метод k-means++
kmeans = KMeans(init="k-means++", n_clusters=unique, n_init=10)
k_alg(kmeans=kmeans, name="k-means++", data=X, labels=Y)

# Метод random
kmeans = KMeans(init='random', n_clusters=unique, n_init=10)
k_alg(kmeans=kmeans, name='random', data=X, labels=Y)

# Метод pca_based
pca = decomposition.PCA(n_components=unique).fit(X)
kmeans = KMeans(init=pca.components_, n_clusters=unique, n_init=10)
k_alg(kmeans=kmeans, name="PCA", data=X, labels=Y)


# Начнем визуализацию наших данных. Сначала применим PCA, далее метод кластеризации KMeans
reduced_X = decomposition.PCA(n_components=2).fit_transform(X)
kmeans = KMeans(init="k-means++", n_clusters=unique, n_init=10)
kmeans.fit(reduced_X)
dataTrainY = kmeans.labels_

# Найдем минимальные и максимальные значения по х и у для задания границ окна
x_min, x_max = reduced_X[:, 0].min() - 1, reduced_X[:, 1].max() + 1
y_min, y_max = reduced_X[:, 0].min() - 1, reduced_X[:, 1].max() + 1

colors = [
          'black', 'firebrick', 'gold', 'chartreuse', 'navy', 
          'm', 'orange', 'yellowgreen', 'violet', 'lightslategray',
         ]

for i in range(unique):
    plt.scatter(reduced_X[dataTrainY==i, 0], reduced_X[dataTrainY==i, 1], c=colors[i], label=str(i))

# Найдем центры кластеров с помощью встроенного метода и отрисуем на графике
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=169,
    linewidth=3,
    color="r",
    zorder=10,
)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.legend(loc=0)
plt.show()
