# Datos No Estructurados: Análisis de Texto



## Identificación de Noticias Fake

A través de este trabajo, mostramos VeriNews, un modelo de procesamiento de lenguaje natural (NLP) diseñado para clasificar noticias como auténticas o falsas. Este trabajo ha sido desarrollado como parte del curso de Datos No Estructurados dentro del Máster en Big Data. Tecnología y Analítica Avanzada de la Universidad Comillas ICAI.

Los componentes de este grupo son:

| Nombre                     | Correo                            |
|----------------------------|-----------------------------------|
| Álvaro Ezquerro Pérez      | alvaroezquerro@alu.comillas.edu   |
| María Calvo de Mora Román  | mcalvodemora@alu.comillas.edu     |
| Celia Quiles Alemañ        | 202315604@alu.comillas.edu        |


### Objetivo y Resumen del trabajo

Nuestro proyecto tiene como objetivo desarrollar un clasificador que pueda identificar con precisión si un titular de noticia es auténtico o falso. Utilizamos un conjunto de datos disponible en Kaggle [1], que consiste en titulares de noticias etiquetados como auténticos o falsos.

Para realizar esta tarea, hemos probado diversas técnicas de procesamiento de lenguaje natural (NLP). De hecho, contamos con 4 notebooks:

1. `LoadingData_and_EDA`: notebook donde analizamos en profundidad los titulares y los textos de las noticias, tanto de noticias reales como fakes.

2. `Preprocess_for_NLP`: en este notebook explicamos paso a paso el preprocesado necesario para aplicar posteriormente modelos de NLP.

3. `NLP_with_MachineLearningModels`: notebook donde comenzamos entrenando modelos simples de Machine Learning (regresión logística, random forest y svm), realizando comparaciones entre ellos.

4. `RNN_and_LSTM`: segundo notebook de modelización, donde se entrenan modelos de Deep Learning (RNN y LSTM).

Finalmente, hemos decidido no incluir un quinto notebook (`Transformers_FineTunning`) principalmente por 2 motivos: por un lado, todos los modelos ejecutados hasta ahora consiguen una accuracy increíblemente positiva. Además, al intentar realizar el fine -tuning se requerían de unos recursos computacionales de los que no disponíamos. En consecuencia, no se han utilizado modelos preentrenados en este proyecto.


### App con Streamlit

Nuestro producto final, VeriNews, es una aplicación web fácil de usar que permite a los usuarios ingresar un titular de noticia y determinar si es auténtico o falso, utilizando nuestro modelo con mejor rendimiento. Proporcionamos instrucciones claras sobre cómo ejecutar la aplicación y obtener el resultado.

VeriNews es una app de Streamlit a través de la cual podremos descubrir si un titular es real o fake.

Streamlit es un marco web de código abierto que permite a los usuarios construir aplicaciones web ligeras para ofrecer capacidades de Machine Learning y Ciencia de Datos utilizando Python. [2]

La app desarrollada es la siguiente:

![FrontEnd_AppVeriNews](images_notebook/FrontEnd_AppVeriNews.png)

Por ejemplo, al introducir un titular procedente del conjunto de Fake News, obtendríamos el siguiente resultado:

![FrontEnd_AppVeriNews](images_notebook/ClassifiedNew_VeriNews.png)



### Referencias

[1] Fake news Detection datasets. (2022, 7 diciembre). Kaggle. https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets

[2] David, D. (2021, 28 octubre). Cómo usar Streamlit y Python para crear una aplicación de ciencia de datos. HackerNoon. https://hackernoon.com/es/como-usar-streamlit-y-python-para-crear-una-aplicacion-de-ciencia-de-datos




