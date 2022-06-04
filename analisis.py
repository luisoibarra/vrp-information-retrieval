# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# # Análisis de los datos
# 
# El objetivo del análisis de los datos es encontrar información acerca del comportamiento de las corridas de problemas con evaluación automática y sin esta.
# 
# ## Datos
# 
# - notape: tiempo sin usar la evaluación automática.
# - tape: tiempo usando la evaluación automática.
# - ratio: tape/notape: el cociente.
# - problem: \<Problema\>-N\<Cantidad De Clientes\>-K\<Rutas Sol Opt\>.
#   - la primera letra es irrelevante.
#   - N: cantidad de clientes.
#   - K: rutas en la solución óptima.
# - criterion: Criterio de selección de vecindad.
#   - RAB: mover un cliente dentro de su ruta.
#   - RARB: mover un cliente para otra ruta (puede ser la misma).
#   - RARAC: interacambiar dos clientes.
#   - REF: mover una subruta dentro de su misma ruta.
#   - RERF: mover una subruta para otra ruta (puede ser la misma).
#   - REREG: intercambiar dos subrutas.
# - routes: cantidad de rutas en la solución inicial del algoritmo.
# - iterations: cantidad de iteraciones que hizo el algoritmo.
# - clients: Cantidad de clientes en el problema.
# - maxroutes: lo puedes ver como si la solución inicial es factible o no.
# - current: esto es para llevar la cuenta de cuánto faltaba.
# - total: el total de repeticiones que se hizo.
# 

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

# %% [markdown]
# ## Lectura de Datos
# 
# Se leyeron los datos y se realizó una limpieza de los mismos:
# - Se eliminaron las corridas que dieron error. (No se encontró ninguna en el conjnto de datos)
# - Se arreglaron los nombres de las columnas y las variables categóricas, ya que tenían espacios al principio o al final
# - Se anotaron otros datos codificados en el nombre del problema como la letra del problema y la cantidad de rutas óptimas
# - Se convirtieron las variables Categóricas en Numéricas.
# - Se eliminaron columnas innecesarias en el análisis
# 

# %%
# Read data
data = pd.read_csv('datos.csv')

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

def plot_hist(data, key, save_fig=None):
    sns.distplot(data[key])
    if save_fig:
        plt.savefig(save_fig)

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

# %% [markdown]
# ## Medidas básicas
# 
# ### Asimetría y Curtosis de ratio
# 
# La asimetría (Skewness) de la variable **ratio** tiene un valor de 1.7. Esto indica que la distribución no es simétrica y que posee una tendencia a dar más valores mayores que menores con respecto a la moda. También indica la posible existencia de outliers superiores.
# 
# La curtosis (Kurtosis) de la variable **ratio** tiene un valor de 15.2. Esto indica que la destribución de los ratios no presenta una desviación estándar grande y que por lo tanto la existencia de outliers no es mucha.
# 
# Estas observaciones se pueden ver en el histograma del **ratio**
# 
# ![images/hist_ratio.png](images/hist_ratio.png)
# 

# %%
# Asimetría y curtosis:

print("Skewness: %f" % data['ratio'].skew())
print("Kurtosis: %f" % data['ratio'].kurt())
plot_hist(data, "ratio", "images/hist_ratio.png")

data.describe()

# %% [markdown]
# ## Correlación
# 
# ![Correlación](images/correlation.png)
# 
# Algunas correlaciones presentes
# 
# **Alta** correlación directa entre:
# 
# - *tape* y *notape*: Ambos representan tiempos de corrida de algoritmos al mismo problema. Esto indica que el análisis en esta matriz relacionados va a ser casi idéntico.
# 
# **Media** correlación directa entre:
# 
# - *iterations* y *clients*: Es lógico pensar que para problemas de mayor espacio de búsqueda la cantidad de iteraciones realiadas por los algoritmos sea mayor.
# - *clients* y *routes*: Lo cual siguiendo la linea de pensamiento anterior, tiene sentido que a mayor cantidad de clientes exista una mayor cantidad de rutas.
# - *routes* y *maxroutes*: Se puede pensar como que empezando por una solución factible (maxroutes=1) inicial los
# algoritmos tienden a hacer más iteraciones que empezando por una solución no factible (maxroutes=0)
# 
# **Débil** correlación inversa entre:
# 
# - *ratio* y *routes*: Al parecer existe una relación no muy marcada entre estos dos atributos que hace que a medida que aumente las rutas en la solución inicial del problema tiende a disminuir el ratio.

# %%
# Todos los datos

corr_data = data.drop("criterion", axis=1)

plot_corr(corr_data, save_fig="images/correlation.png")
plot_corr_scatter_matrix(corr_data, corr_data.columns, save_fig="images/correlation_scatter.png")

# %% [markdown]
# ## Análisis Outliers
# 
# Se eliminaron los outliers del atributo ratio por el método de IQR, estos datos representan solamente 15 entradas siendo todos superiores, no llegando al 1% de la muestra.
# 
# En los outliers se muestra la siguiente tabla de correlación, aunque esta no tiene mucha significación estadística por la poca cantidad de elementos que la conforman da a conocer información sobre estos.
# 
# ![images/correlation_upper_ratio_outliers.png](images/correlation_upper_ratio_outliers.png)
# 
# En estos datos aparecieron nuevas correlaciones con respecto al ratio, aunque débiles, por ejemplo la correlación inversa entre el ratio y las iterations.
# 
# Eliminando los outliers de la muestra la distribución de los ratios se hace normal, según la prueba de Shapiro-Wilks.

# %%
# Outliers

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

print("All data")
test_normal_dstribution(data, "ratio")
print("No outliers")
test_normal_dstribution(no_ratio_outliers, "ratio")

plot_corr(no_ratio_outliers, save_fig="images/correlation_no_ratio_outliers.png")
plot_corr_scatter_matrix(no_ratio_outliers, no_ratio_outliers.columns, save_fig="images/correlation_scatter_no_ratio_outliers.png")

plot_corr(upper_ratio_outliers, save_fig="images/correlation_upper_ratio_outliers.png")
plot_corr_scatter_matrix(upper_ratio_outliers, corr_data.columns, save_fig="images/correlation_scatter_upper_ratio_outliers.png")

# %% [markdown]
# ## Análisis de ratio <= 1
# 
# Dado que los que cumplen este criterio cumplen que la corrida con evaluación automática fue mejor o igual que la corrida sin esta tiene significancia ver que propiedades cumplen este grupo.
# 
# En este grupo se encontraron 40 entradas, lo que representa un 1.6% de la muestra.
# 
# Entre las características que presentan dichas corridas se encuentran:
# 
# - La mayoría de los ratios se encuentra próximos a 1.
# - La mayoría empezó con una solución inicial factible.
# - La mayoría se realizaron en menos de 25 iteraciones.
# - La mayoría se realizaron con menos de 50 clientes.
# - El criterio 0 (TODO poner nombre) fue el más observado.
# 
# ![images/correlation_scatter_ratio_lower_than1.png](images/correlation_scatter_ratio_lower_than1.png)
# 

# %%
low_ratio_data = data[data["ratio"] <= 1]

print(low_ratio_data.describe())

plot_corr_scatter_matrix(low_ratio_data, low_ratio_data.columns, save_fig="images/correlation_scatter_ratio_lower_than1.png")

# %% [markdown]
# ## Análisis de clients
# 
# Se observa un ligero comportamiento decreciente en el **ratio** a medida que aumenta la cantidad de clientes.
# Aunque solamente se tiene medida de 32, 33, 37, 65, 80, 135 clientes respectivamente.
# 
# ![images/boxplot_ratio_groupedby_clients.png](images/boxplot_ratio_groupedby_clients.png)
# 
# ![images/mean_decreasing_ratio.png](images/mean_decreasing_ratio.png)
# 

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
# ## Análisis de criterion
# 
# El análisis del criterio indica que individualmente no influye en el ratio. Se realizó un test de ANOVA sobre colecciones con el mismo criterio dando como resultado que este no tenía influencia sobre el valor del ratio.
# 
# Aumentando nuestro análisis sobre este tema se investigó si la cantidad de clientes y el criterio afectaban el radio llegando a la conclusión de que iba disminuyendo la media a medida que aumentaba la cantidad de clientes, reforzando lo mostrado anteriormente:
# 
# ![images/boxplot_ratio_groupedby_criterion_cients.png](images/boxplot_ratio_groupedby_criterion_cients.png)
# 
# En la imagen se observa en el eje x una tupla que representa (criterio, cantidad de clientes) por donde se filtraron los ratios para hallarle el boxplot. Se observa un leve descenso con el aumento de los clientes.
# 

# %%
for cr in set(data["criterion"]):
    print("testing anova with criterion", cr)
    anova_data = data[data["criterion"] == cr]
    print(len(anova_data))
    test_anova(anova_data, "criterion", "ratio")

plot_grouped_by_boxplot(data, "ratio", ["criterion", "clients"], save_fig="images/boxplot_ratio_groupedby_criterion_cients.png", figsize=(22,3))

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

keys = ["routes", "iterations", "clients"]

# Plotting elbow 
plot_elbow(non_obj[keys])

# Elbow method shows that 7-8 clusters are a good choice
labeled_normalized_data = kmeans(non_obj, 7)


# %%
# Plotting results
# TODO Try with different choices to see what happens
plot_kmeans_clusters(labeled_normalized_data, keys)


