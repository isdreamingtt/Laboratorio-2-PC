# Biblical Text Mining - Laboratorio 2

Proyecto desarrollado para el Laboratorio 2 de Programación Científica. El sistema implementa un pipeline de análisis computacional sobre un corpus bíblico en inglés, utilizando técnicas de procesamiento de lenguaje natural, representación vectorial, visualización, clasificación, generación de texto y análisis de sentimiento.

## Autores

* Harold Ramos
* Cristian Perez

## Descripción general

El proyecto trabaja con una versión estructurada de la Biblia, donde la unidad principal de análisis es el versículo. A partir del texto original, el sistema realiza preprocesamiento, construye representaciones TF-IDF, calcula similitud semántica, genera visualizaciones, entrena un clasificador de versículos, produce texto mediante modelos n-grama y analiza el sentimiento por libro y capítulo.

El sistema fue desarrollado en Python utilizando programación orientada a objetos. Las clases principales están separadas por responsabilidad para mantener una estructura clara y modular.

## Funcionalidades principales

El sistema incluye las siguientes funcionalidades:

1. Preprocesamiento de texto.
2. Construcción de vocabulario y cálculo de frecuencias.
3. Implementación manual de TF-IDF.
4. Implementación manual de similitud del coseno.
5. Motor de búsqueda semántico.
6. Visualización y análisis exploratorio del corpus.
7. Heatmap de similitud entre libros.
8. Visualización de versículos usando PCA.
9. Clasificador de versículos por libro.
10. Generador de texto basado en unigramas, bigramas, trigramas y cuatrigramas.
11. Análisis de sentimiento por versículo, capítulo y libro.
12. Generación de gráficos y archivos procesados.

## Estructura del proyecto

```text
Laboratorio 2 PC/
│
├── datos/
│   ├── dataset_original/
│   │   ├── t_bbe.csv
│   │   ├── key_english.csv
│   │   └── bible_version_key.csv
│   │
│   └── dataset_procesado/
│
├── outputs/
│   └── graficos/
│
├── src/
│   ├── main.py
│   ├── preprocesamiento.py
│   ├── tfidf.py
│   ├── similitud_coseno.py
│   ├── procesador_semantico.py
│   ├── visualizador.py
│   ├── clasificador.py
│   ├── generador_ngramas.py
│   └── analisis_sentimiento.py
│
├── reportes/
│   ├── Diagrama de clases Taller 2 PC.pdf
|   └── Informe Laboratorio 2 Programación Científica.pdf
│
├── requirements.txt
└── README.md
```

## Clases principales

### `Preprocesador`

Clase encargada del procesamiento inicial del texto. Realiza limpieza, tokenización, eliminación de stopwords, construcción de vocabulario y cálculo de frecuencias.

### `CalculadorTFIDF`

Clase que implementa manualmente el cálculo de TF-IDF. Incluye métodos para calcular TF, DF, IDF y vectores TF-IDF por documento.

### `SimilitudCoseno`

Clase encargada de calcular la similitud del coseno entre dos vectores. Implementa producto punto, magnitud vectorial y cálculo de similitud.

### `BuscadorSemantico`

Clase que permite ingresar una consulta y obtener los K versículos más similares del corpus. Utiliza TF-IDF y similitud del coseno.

### `VisualizadorCorpus`

Clase encargada de generar gráficos exploratorios del corpus, incluyendo cantidad de versículos por libro, palabras frecuentes, longitud promedio de versículos, heatmap de similitud entre libros y PCA.

### `ClasificadorVersiculos`

Clase que entrena un modelo de clasificación para predecir el libro al que pertenece un versículo. Utiliza TF-IDF como representación de entrada y regresión logística como modelo de clasificación.

### `GeneradorNGramas`

Clase encargada de entrenar modelos de lenguaje probabilísticos basados en unigramas, bigramas, trigramas y cuatrigramas. Permite generar secuencias de texto desde una palabra inicial.

### `AnalizadorSentimiento`

Clase que calcula puntajes de sentimiento por versículo usando VADER. Luego agrega los resultados por libro y capítulo, genera gráficos y muestra los casos más positivos y negativos.

## Instalación

Primero se recomienda clonar o descargar el repositorio.

Luego, desde la carpeta principal del proyecto, instalar las dependencias con:

```bash
pip install -r requirements.txt
```

El archivo `requirements.txt` contiene las librerías necesarias para ejecutar el sistema:

```text
matplotlib==3.10.9
pandas==2.3.3
scikit-learn==1.7.2
vaderSentiment==3.3.2
```

## Ejecución

Desde la carpeta principal del proyecto, ejecutar:

```bash
python src/main.py
```

El programa mostrará un menú interactivo con las siguientes opciones:

```text
1. Ver información del corpus original
2. Ver información del corpus resultante
3. Buscar semántico
4. Visualización y análisis exploratorio
5. Clasificador de versículos
6. Generador de texto
7. Análisis de sentimiento
0. Salir
```

## Uso del sistema

### 1. Información del corpus original

Muestra información general del dataset original, incluyendo primeras filas, cantidad de versículos, cantidad de libros y distribución por testamento.

### 2. Información del corpus resultante

Muestra el corpus preprocesado, cantidad de palabras únicas, palabras más frecuentes y permite guardar archivos procesados en formato CSV.

### 3. Buscador semántico

Permite ingresar una frase y obtener los versículos más similares del corpus. El resultado incluye libro, capítulo, versículo, texto original y valor de similitud.

### 4. Visualización y análisis exploratorio

Genera gráficos relacionados con la estructura del corpus:

* Cantidad de versículos por libro.
* Palabras más frecuentes.
* Longitud promedio de versículos por libro.
* Heatmap de similitud entre libros.
* PCA de versículos usando TF-IDF.

Los gráficos se guardan en:

```text
outputs/graficos/
```

### 5. Clasificador de versículos

Entrena un modelo de clasificación para predecir el libro al que pertenece cada versículo. El sistema divide los datos en entrenamiento y prueba, entrena el modelo, calcula accuracy y genera una matriz de confusión.

### 6. Generador de texto

Entrena modelos de n-gramas y permite generar texto desde una palabra inicial ingresada por el usuario. El sistema compara la salida de:

* Unigrama.
* Bigrama.
* Trigrama.
* Cuatrigrama.

### 7. Análisis de sentimiento

Calcula sentimiento por versículo y luego agrega los resultados por libro y capítulo. También genera gráficos de sentimiento y muestra los libros y capítulos con valores más positivos y negativos.

Los archivos generados se guardan en:

```text
datos/dataset_procesado/
```

## Archivos generados

Durante la ejecución del sistema se pueden generar archivos CSV y gráficos.

### Archivos CSV

```text
datos/dataset_procesado/corpus_preprocesado.csv
datos/dataset_procesado/frecuencias_palabras.csv
datos/dataset_procesado/corpus_sentimiento.csv
datos/dataset_procesado/sentimiento_por_libro.csv
datos/dataset_procesado/sentimiento_por_capitulo.csv
```

### Gráficos

```text
outputs/graficos/versiculos_por_libro.png
outputs/graficos/palabras_frecuentes.png
outputs/graficos/longitud_promedio_por_libro.png
outputs/graficos/heatmap_similitud_libros.png
outputs/graficos/pca_versiculos.png
outputs/graficos/matriz_confusion.png
outputs/graficos/sentimiento_por_libro.png
outputs/graficos/sentimiento_por_capitulo.png
```

## Diagrama de clases

El proyecto incluye un diagrama de clases UML donde se muestran las clases principales, sus atributos, métodos y relaciones.

Archivo:

```text
reportes/Diagrama de clases Taller 2 PC.pdf
```

## Informe

El análisis detallado de resultados, decisiones técnicas, limitaciones y conclusiones se encuentra en el informe del laboratorio.

Archivo:

```text
reportes/Informe Laboratorio 2 Programación Científica.pdf
```
