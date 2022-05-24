# Orientación

Se da un problema y este es resuelto mediente dos algoritmos dando una tabla de resultados. El objetivo es sacar información sobre las corridas de estos en base del tiempo que se demoraron para ver si existe alguna relación.

## Datos

- notape: Tiempo de corrida sin grafo?
- tape: Tiempo de corrida con grafo?
- ratio: tape/notape
- problem: Tipo de problema, tiene en su nombre la cantidad de clientes N-33/
- criterion:
- routes: Cantidad de rutas que se usaron en el problema
- iterations:
- clients: Cantidad de clientes en el problema?
- maxroutes: Variable que indica si se hizo algo o no.
- current:
- total:

## Qué hizo Fernando

- Leyó los datos y el tiempo, no tenemos el tiempo
- Filtró las entradas con error
- Observó las estadísticas descriptivas
- Observó los outliers usando IQR
- Observó la correlación entre Tiempo sin grafo (notape) y Tiempo con grafo (tape)
- Observó el ratio (tape/notape)
- Hace una regresión lineal entre notape y tape usando sklearn.
- Observa los datos con outliers y sin outliers en el ratio

## A Completar

- [ ] Que significa la K en el nombre del problema? A-N22-K5
- [ ] Preguntar por los campos uque significan


## Notas

### IQR

Inter-Quantile-Rank (Rango intercuantílico)
$$
IQR = Q_3 - Q_1
$$

- <https://www.universoformulas.com/estadistica/descriptiva/rango-intercuartilico/>

Es la diferencia entre el tercer cuantil y el primer cuantil. Se interpreta como una medida de dispersión de los datos. Es usada generalmente en distribuciones asimétricas en donde es meor medir la dispersión con respecto a la mediana.

El IQR se usa también para detectar outliers en los datos <https://towardsdatascience.com/practical-implementation-of-outlier-detection-in-python-90680453b3ce>.
