# Orientación

Se da un problema y este es resuelto mediente dos algoritmos dando una tabla de resultados. El objetivo es sacar información sobre las corridas de estos en base del tiempo que se demoraron para ver si existe alguna relación.

## Datos

- notape: tiempo sin usar la evaluación automática
- tape: tiempo usando la evaluación automática
- ratio: tape/notape: el cociente
- problem: \<Problema\>-N\<Cantidad De Clientes\>-K\<??\>
  - la primera letra es irrelevante
  - N: cantidad de clientes
  - K: rutas en la solución óptima
- criterion: Criterio de selección de vecindad
  - RAB: mover un cliente dentro de su ruta
  - RARB: mover un cliente para otra ruta (puede ser la misma)
  - RARAC: interacambiar dos clientes
  - REF: mover una subruta dentro de su misma ruta
  - RERF: mover una subruta para otra ruta (puede ser la misma)
  - REREG: intercambiar dos subrutas
- routes: cantidad de rutas en la solución inicial del algoritmo.
- iterations: cantidad de iteraciones que hizo el algoritmo
- clients: Cantidad de clientes en el problema
- maxroutes: lo puedes ver como si la solución inicial es factible o no 🤷🏻‍♀️
- current: esto es para llevar la cuenta de cuánto faltaba 🥵
- total: el total de repeticiones que se hizo.

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

- [ ] Poner los resultados de los .describe() en algún lado como valor extra
- [ ] Hacer KNN
  - Fijarse en routes con iterations
- [ ] Hacer dendrograma
- [ ] Mejorar Analisis de outliers

## Notas

### IQR

Inter-Quantile-Rank (Rango intercuantílico)
$$
IQR = Q_3 - Q_1
$$

- <https://www.universoformulas.com/estadistica/descriptiva/rango-intercuartilico/>

Es la diferencia entre el tercer cuantil y el primer cuantil. Se interpreta como una medida de dispersión de los datos. Es usada generalmente en distribuciones asimétricas en donde es meor medir la dispersión con respecto a la mediana.

El IQR se usa también para detectar outliers en los datos <https://towardsdatascience.com/practical-implementation-of-outlier-detection-in-python-90680453b3ce>.
