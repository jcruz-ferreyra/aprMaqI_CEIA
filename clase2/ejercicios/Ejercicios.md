# Ejercicios para prácticar

1. Usando el dataset `winequality-red.csv`, el cual consiste en datos de vinos rojos basados en datos físico-químicos y una métrica de calidad de vino. Más info en [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009). Queremos predecir la calidad del vino usando los atributos físico-químicos del mismo.
	1. Lea el dataset como un DataFrame de Pandas. Realice un estudio de variables. Como se llaman y que están midiendo exactamente (vea la documentación del dataset). Además, analice que tipo de variables (incluido el target) son, cuál es el rango de estas variables y cómo se distribuyen (histograma). Además, realice una matriz de correlación, ¿cuáles variables parecen estar correlacionadas? y con respectos a la calidad del vino?
	2. Realice si es necesario limpieza de datos y corrección de errores.
	3. Construya un modelo de regresión lineal simple, el cual se intente predecir la calidad del vino usando el nivel de alcohol.
    	1. Realiza la separación entre el dataset de entrenamiento y testeo. Utilice 80%-20%.
    	2. Determine que métrica se va a usar para evaluar la calidad del modelo (MSE, MAE, etc.)
    	3. Entrene el modelo con el set de entrenamiento.
    	4. Evalúe el modelo con la métrica de evaluación y el coeficiente de Pearson.
	4. Construya un modelo de regresión lineal múltiple, usando todos los atributos
    	1. Realiza la separación entre el dataset de entrenamiento y testeo. Utilice 80%-20%.
    	2. Entrene el modelo con el set de entrenamiento.
    	3. Evalúe el modelo con la métrica de evaluación y el coeficiente de Pearson. Use la misma métrica que en el punto anterior.
	5. Partiendo del modelo anterior, que involucra todos los atributos, realiza una construcción de modelo hacia atrás, de la misma forma que vimos en clase (utilice `statsmodels`) y el test de hipótesis de bondad de ajuste hasta que se llegue el número de atributos que no se puedan eliminar.
    	1. Una vez seleccionado los atributos a usar, entrene el modelo y realice los pasos del punto anterior.
	6. En función de los resultados obtenidos, discuta los modelos, y que tan bien explican la calidad del vino.

2. Lea el dataset `bluegills.csv` (OBS, tenga en cuenta que el csv la separación entre columnas es con **\t**), el cual consiste en el registro de 78 lepomis macrochirus medidos en longitud y edad. Se busca determinar como el tamaño del pez depende de la edad. Mas info [acá](https://online.stat.psu.edu/stat501/lesson/9/9.8). Realice un analisis inicial de los datos y construya modelos de regresion polinomica de diferente orden, haciendo previamente la separación entre datos de entrenamiento y validación. Mida con alguna metrica y encuentre el modelo que mejor se ajusta a los datos de validación.
