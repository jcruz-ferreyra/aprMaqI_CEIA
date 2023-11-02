# Ejercicios para prácticar

Sos un data scientist que trabaja para una empresa de publicidad que lanzo una campaña de publicidad en una red social. Se registró cada usuario al que se le mostró la publicidad, y se pudo obtener datos de la persona tales como `genero`, `edad` y `salario estimado`. Además se registró si el usuario luego compró el producto que la publicidad hacia referencia. Los datos están en `Social_Network_Ads.csv`. Se busca poder predecir dado un usuario con datos socioseconomicos si va a comprar o no el producto.

1. Realice un estudio de variables y de limpieza de datos. Analize las clases, estan balanceadas, o no? Qué clase nos parece mas importante de las dos?
2. Separe el dataset en entrenamiento y validación.
3. Elija diferente modelos de clasificación (al menos uno de regresión logistica y uno de KNN). Elija las variables de entrada en base al analisis del punto 1.
4. Comparelos con dos o más metrica de evaluación. Cual fue el mejor modelo? Todas las metricas coincidieron o metricas diferentes evaluan como mejor a diferentes modelos? Discuta los resultados.
5. Cree curvas ROC para evaluar el modelo para ver la calidad del modelo, sin depender del valor umbral. Elija un valor umbral que considere más optimo y vuelva a clasificar usando ese valor. Como cambiaron las metricas usadas en el punto 4 con este valor umbral.