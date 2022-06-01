# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# Importo las bibliotecas básicas:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')  # Para evitar los molestos avisos.
get_ipython().run_line_magic('matplotlib', 'inline')


# %%
# Read data
data = pd.read_csv('datos.csv')


# %%
# Clean data

# Problems with errors
zero_time = np.where(data['notape'] <= 1e-17)
print("Datos con errores")
print(zero_time)
data.drop(zero_time[0], inplace = True)

# Fixing columns names
data.rename(columns={
    key: key.strip() for key in data.keys()
    }, inplace = True)

# Fixing string values
for col in data.columns:
    if type(data[col][0]) == str:
        data[col] = [val.strip() for val in data[col]]
        
        
# Adding problem label
data['problem_tag'] = data['problem'].apply(lambda x: x[0])
data['optim_routes'] = data['problem'].apply(lambda x: int(x[x.index("K")+1:]))


# %%
# Mapear las variables categóricas a números
categoricals_columns= ['problem','criterion']

# fetch all values in problem column
problem_values = list(set(data['problem'].values))
criterion_values = list(set(data['criterion'].values))

problem_values.sort()
criterion_values.sort()

# Por cada valor cambiarlo en el dataframe por su indice en la lista
for problem in problem_values:
    data['problem'].replace(problem, problem_values.index(problem), inplace=True)
for criterion in criterion_values:
    data['criterion'].replace(criterion, criterion_values.index(criterion), inplace=True)

# Ordenar las columnas con 'tape' , 'notape' y 'ratio' como primeras
columns = data.columns.values
columns = np.append(['tape','notape','ratio'], columns[~np.in1d(columns, ['tape','notape','ratio'])])
data = data[columns]

# Eliminar columnas innecesarias para el analisis
data.drop(columns=["current", "total"], inplace=True) # Current y total no se consideran necesarias

# Convertir las variables categóricas en variables ficticias o dummies:

# data = pd.get_dummies(data)


# %%
def describe(data):
    print(data.describe())
    print(data.info())
    
    plot_boxplot(data, 'ratio')

def plot_boxplot(data, key:str):
    """
    Plot a boxplot using the dataframe indexing on key
    """
    plt.boxplot(data[key])
    plt.title(f"Box Plot {key}")
    plt.legend([key])
    plt.show()

def plot_hist(data, key):
    sns.distplot(data[key])

def plot_qq(data, key):
    stats.probplot(data[key], plot=plt)
    plt.show()

def plot_corr_scatter_matrix(data, keys, save_fig=None):
    sns.set()
    sns.pairplot(data[np.array(keys)], size=2.5)
    if save_fig:
        plt.savefig(save_fig)
    plt.show()

def iqr(data, key):
    """
    Calculates the IQR from the data associated with key
    """
    Q1 = np.percentile(data[key], 25,
                interpolation = 'midpoint')
  
    Q3 = np.percentile(data[key], 75,
                    interpolation = 'midpoint')
    IQR = Q3 - Q1
    return Q1, Q3, IQR

def get_outliers(data, key, upper=True):
    """
    Returns the outliers of data[key]
    """
    Q1, Q3, IQR = iqr(data,key)
    if upper:
        return data[data[key] >= Q3 + IQR*1.5]
    return data[data[key] <= Q1 - IQR*1.5]

def remove_outliers(data, key, remove_upper=True, remove_lower=True):
    """
    Returns the data in key without the outliers 
    """
    base_data = data
    if remove_upper:
        base_data = base_data[~base_data.isin(get_outliers(data, key))]
    if remove_lower:
        base_data = base_data[~base_data.isin(get_outliers(data, key, False))]
    return base_data

def plot_corr(df, save_fig=None, size=10):
    """
    Function plots a graphical correlation matrix
    for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """

    corr = df.corr()
    corr.style.background_gradient(cmap='coolwarm').set_precision(2)
    fig, ax = plt.subplots(figsize=(size+size/2, size))
    
    # ax.matshow(corr)
    # plt.xticks(range(len(corr.columns)), corr.columns)
    # plt.yticks(range(len(corr.columns)), corr.columns)
    
    sns.heatmap(corr,
            cmap='coolwarm',
            annot=True,
            )
    
    if save_fig:
        plt.savefig(save_fig)

def plot_scatter_matrix(data, keys:tuple, save_fig=None):
    pd.plotting.scatter_matrix(data.loc[:,keys])
    if save_fig:
        plt.savefig(save_fig)
    
def plot_grouped_by_boxplot(data, objective, groupby_keys, save_fig=None, **kwargs):
    df = pd.DataFrame({str(k): value[objective] for k,value in data[groupby_keys + [objective]].groupby(groupby_keys)})
    df.plot(kind='box', title=f"{objective} boxplot grouped by {', '.join(groupby_keys)}", **kwargs)
    if save_fig:
        plt.savefig(save_fig)
    plt.show()
    
def test_anova(data, factor, objective, alpha=0.1):
    import scipy.stats as stats
    levels = set(data[factor])
    levels = [data[objective][data[factor] == level] for level in levels]
    result = stats.f_oneway(*levels)
    if result.pvalue < alpha:
        # H0 is rejected
        print(f"ANOVA: {factor} influences {objective}")
    else:
        print(f"ANOVA: {factor} does not influences {objective}")
    print(result.pvalue)

def test_normal_dstribution(data, key, alpha=0.1):
    _, pvalue = stats.shapiro(data[key])
    if pvalue > alpha:
        print(f"Shappiro test on {key}: Probably Gaussian with pvalue {pvalue}")
        return True
    else:
        print(f"Shappiro test on {key}: Probably NOT Gaussian with pvalue {pvalue}")
        return False


# %%
# Asimetría y curtosis:

print("Skewness: %f" % data['ratio'].skew())
print("Kurtosis: %f" % data['ratio'].kurt())

# %% [markdown]
# ## Correlación

# %%
# Todos los datos

corr_data = data.drop("criterion", axis=1)

plot_corr(corr_data, save_fig="images/correlation.png")
plot_corr_scatter_matrix(corr_data, corr_data.columns, save_fig="images/correlation_scatter.png")

# %% [markdown]
# ## Outliers

# %%
# Outliers TODO

corr_data = data.drop("criterion", axis=1)

## ratio
upper_ratio_outliers = get_outliers(corr_data, 'ratio')
lower_ratio_outliers = get_outliers(corr_data, 'ratio', False)
no_ratio_outliers = remove_outliers(corr_data, "ratio")

print("Upper ratio outliers")
print(upper_ratio_outliers.describe())

print("Lower ratio outliers")
print(lower_ratio_outliers.describe())

print("No ratio outliers")
print(no_ratio_outliers.describe())

test_normal_dstribution(data, "ratio")

plot_corr(no_ratio_outliers, save_fig="images/correlation_no_ratio_outliers.png")
plot_corr_scatter_matrix(no_ratio_outliers, no_ratio_outliers.columns, save_fig="images/correlation_scatter_no_ratio_outliers.png")

plot_corr(upper_ratio_outliers, save_fig="images/correlation_upper_ratio_outliers.png")
plot_corr_scatter_matrix(upper_ratio_outliers, corr_data.columns, save_fig="images/correlation_scatter_upper_ratio_outliers.png")


# %%
# No outliers TODO

# corr_data = data.drop("criterion", axis=1)

# plot_corr(corr_data, save_fig="images/correlation.png")
# plot_corr_scatter_matrix(corr_data, corr_data.columns, save_fig="images/correlation_scatter.png")

# %% [markdown]
# # Clustering
# 
# Trying K-Mean to make clusters and visualize data

# %%
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score
 

def plot_elbow(data, cluster_range:tuple=(2,20)):
    inertias = []
    for i in range(*cluster_range):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        # print(f"Silhouette with {i} clusters: {silhouette_score(data, kmeans.labels_)}")
    
    plt.plot(range(*cluster_range), inertias)
    plt.title("Elbow curve")
    plt.show()

def kmeans(data, clusters:int, x_label=None, y_label=None):
    """
    Returns the scaled data annotaed with clusters labels
    """
    scaler = MinMaxScaler() # StandardScaler()
    scale = scaler.fit_transform(data)
    scale = pd.DataFrame(scale, columns=data.columns)
    
    model = KMeans(n_clusters=clusters) # DBSCAN(eps=??, min_samples=??)
    clusters = model.fit_predict(scale)
    
    scale["clusters"] = clusters
    return scale

def plot_kmeans_clusters(data, labels:tuple):
    """
    Plots the `labels` annotated with `clusters` in data.
    """
    if len(labels) == 2:
        sns.scatterplot(x=labels[0], y=labels[1], hue = 'clusters',  data=data, palette='viridis')
        plt.show()
    if len(labels) == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel(labels[2])
        ax.set_title("Cluster of " + ", ".join(labels))

        scat_plot = ax.scatter(data[labels[0]],data[labels[1]],data[labels[2]], c=data['clusters'])
        plt.show()

non_obj = data.drop([col for col in data.columns if data[col].dtype == 'O'], axis=1)

# Plotting elbow 
plot_elbow(non_obj[keys])

# Elbow method shows that 7-8 clusters are a good choice
labeled_normalized_data = kmeans(non_obj, 7)


# %%
# Plotting results
# TODO Try with different choices to see what happens
keys = ["routes", "iterations", "clients"]
plot_kmeans_clusters(labeled_normalized_data, keys)

# %% [markdown]
# ## ratio <= 1

# %%
low_ratio_data = data[data["ratio"] <= 1]

print(low_ratio_data.describe())

plot_corr_scatter_matrix(low_ratio_data, low_ratio_data.columns, save_fig="images/correlation_scatter_ratio_lower_than1.png")

# %% [markdown]
# ## Client Analysis

# %%
# Grouping by number of clients and computing the mean for each group
remove_out = remove_outliers(data, "ratio")

computed = remove_out[["clients", "tape", "notape", "ratio"]].groupby(["clients"]).mean()
print(computed.index)
plt.plot(computed.index, computed["ratio"], label="ratio")
plt.xlabel("clients")
plt.ylabel("mean of ratio")
plt.legend()
plt.savefig("images/mean_decreasing_ratio.png")
plt.show()
plt.plot(computed.index, computed["tape"], label="tape")
plt.plot(computed.index, computed["notape"], label="notape")
plt.xlabel("clients")
plt.ylabel("mean of tape/notape")
plt.legend()
plt.savefig("images/mean_running_time.png")
plt.show()

plot_grouped_by_boxplot(remove_out, "ratio", ["clients"], save_fig="images/boxplot_ratio_groupedby_clients")
# plot_grouped_by_boxplot(data, "routes", ["clients"])
# plot_grouped_by_boxplot(data, "tape", ["clients"])
# plot_grouped_by_boxplot(data, "notape", ["clients"])

plt.scatter(remove_out["clients"], remove_out["ratio"])
plt.show()

# %% [markdown]
# ## Criterion analysis

# %%
for cr in set(data["criterion"]):
    print("testing anova with criterion", cr)
    anova_data = data[data["criterion"] == cr]
    print(len(anova_data))
    test_anova(anova_data, "criterion", "ratio")

plot_grouped_by_boxplot(data, "ratio", ["criterion", "clients"], save_fig="images/boxplot_ratio_groupedby_criterion_cients.png", figsize=(22,3))


