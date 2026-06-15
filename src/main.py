import pandas as pd
from preprocesamiento import Preprocesador
from tfidf import CalculadorTFIDF
from similitud_coseno import SimilitudCoseno
from procesador_semantico import BuscadorSemantico
from visualizador import VisualizadorCorpus

def cargar_corpus():
    print("\nCargando datasets originales...")

    biblia = pd.read_csv("datos/dataset_original/t_bbe.csv")
    libros = pd.read_csv("datos/dataset_original/key_english.csv")

    biblia = biblia.rename(columns={
        "field": "id",
        "field.1": "id_libro",
        "field.2": "capitulo",
        "field.3": "versiculo",
        "field.4": "texto"
    })

    libros = libros.rename(columns={
        "field": "id_libro",
        "field.1": "libro",
        "field.2": "testamento",
        "field.3": "id_genero"
    })

    corpus = pd.merge(biblia, libros, on="id_libro", how="left")
    #Se hizo un merge para unir ambos datasets, usando la columna "id_libro" como clave. Se usó un merge de tipo "left" 
    #para mantener todos los versículos de la biblia, incluso si no tienen información de libro en el dataset de libros.

    corpus = corpus[
        ["id", "id_libro", "libro", "testamento", "id_genero", "capitulo", "versiculo", "texto"]
    ]

    print("Corpus base cargado correctamente.")

    return corpus

def infoDatasetOriginal(corpus):
    print("\nPrimeras 5 filas del corpus base:")
    print(corpus.head())

    print("\nCantidad total de versículos:")
    print(len(corpus))

    print("\nCantidad total de libros:")
    print(corpus["libro"].nunique())

    print("\nCantidad por testamento:")
    print(corpus["testamento"].value_counts())

def infoDatasetPreprocesado(corpus, estado):

    vocabulario = estado["vocabulario"]
    frecuencias_df = estado["frecuencias_df"]

    print("\nPrimeras 5 filas del corpus preprocesado:")
    print(corpus.head())

    print("\nCantidad total de versículos:")
    print(len(corpus))

    print("\nCantidad total de libros:")
    print(corpus["libro"].nunique())

    print("\nCantidad por testamento:")
    print(corpus["testamento"].value_counts())

    print("\nCantidad de palabras únicas en el vocabulario:")
    print(len(vocabulario))

    print("\n10 palabras más frecuentes:")
    print(frecuencias_df.head(10))

    opcion = input("\n¿Desea guardar el corpus preprocesado y las frecuencias en nuevos archivos CSV? (s/n): ")
    if opcion.lower() == "s":
        corpus.to_csv("datos/dataset_procesado/corpus_preprocesado.csv", index=False)
        frecuencias_df.to_csv("datos/dataset_procesado/frecuencias_palabras.csv", index=False)
        print("Corpus preprocesado guardado como 'datos/dataset_procesado/corpus_preprocesado.csv'.")
        print("Frecuencias guardadas como 'datos/dataset_procesado/frecuencias_palabras.csv'.")

def procesar_corpus(corpus_original):

    corpus_procesado = corpus_original.copy()

    preprocesador = Preprocesador()
    
    textos_limpios = []
    tokens_procesados = []
    lista_tokens = []

    for texto in corpus_original["texto"]:
        texto_limpio, tokens = preprocesador.preprocesar_texto(texto)

        textos_limpios.append(texto_limpio)
        tokens_procesados.append(" ".join(tokens))
        lista_tokens.append(tokens)

    corpus_procesado["texto_limpio"] = textos_limpios
    corpus_procesado["tokens"] = tokens_procesados

    vocabulario, frecuencias = preprocesador.construir_vocabulario_y_frecuencias(lista_tokens)
    frecuencias_df = pd.DataFrame(list(frecuencias.items()), columns=["palabra", "frecuencia"])
    frecuencias_df = frecuencias_df.sort_values("frecuencia", ascending=False).reset_index(drop=True)

    return corpus_procesado, lista_tokens, vocabulario, frecuencias_df

def buscador_semantico(estado):

    corpus = estado["corpus"]
    lista_tokens = estado["lista_tokens"]
    preprocesador = estado["preprocesador"]

    print("\nTF-IDF test")

    calculador_tfidf = estado["calculador_tfidf"]
    
    matriz_tfidf = calculador_tfidf.calcular_tfidf_corpus(lista_tokens)

    print("\nCantidad de documentos vectorizados:", len(matriz_tfidf))
    print("\nCantidad de palabras con IDF:", len(calculador_tfidf.idf))

    print("\nTop 10 palabras TF-IDF del primer versículo:") #print de ejemplo, 
    top_primer_versiculo = calculador_tfidf.obtener_top_tfidf(matriz_tfidf[0], 10)

    for palabra, valor in top_primer_versiculo:
        print(palabra, ":", valor)

    #Calculo de tf-idf para el primer versículo, solo es una prueba para verificar que el proceso se realizó correctamente.
    
    print("\nSimilitud del coseno --------------------------------------")

    similitud_coseno = estado["similitud_coseno"]

    similitud_versiculos = similitud_coseno.calcular_similitud( #prueba para ver si funciona 
        matriz_tfidf[0],
        matriz_tfidf[1]
    )

    print("\nSimilitud entre el versiculo 1 y el versiculo 2:") #prueba para ver si funciona
    print(similitud_versiculos)


    print("\nBuscador semántico---------------------------------------")

    buscador = BuscadorSemantico(
        corpus,
        matriz_tfidf,
        preprocesador,
        calculador_tfidf
    )

    consulta = "God created the heaven and the earth" #consulta definida fija para probar el buscador semántico, 
                                                      #se espera hacerlo más manual para el usuaorio en el futuro, pero por 
                                                      #ahora se define directamente en el código para verificar que el proceso funciona 
                                                      #correctamente.
    resultados = buscador.buscar(consulta, 5)

    print("\nConsulta:")
    print(consulta)

    print("\nListado de 5 versículos más similares:")

    for resultado in resultados:
        print("--------------------------------------")
        print("Libro:", resultado["libro"])
        print("Capítulo:", resultado["capitulo"])
        print("Versículo:", resultado["versiculo"])
        print("Similitud:", resultado["similitud"])
        print("Texto:", resultado["texto"])

def visualizacion_analisis_exploratorio(estado):
    corpus = estado["corpus"]
    frecuencias_df = estado["frecuencias_df"]

    print("\nVisualización y análisis exploratorio")
    print("--------------------------------------")

    visualizador = VisualizadorCorpus()
    visualizador.generar_visualizaciones_basicas(corpus, frecuencias_df)

    print("\nVisualizaciones básicas generadas correctamente.")

def main():

    corpus_original = cargar_corpus()
    corpus_preprocesado, lista_tokens, vocabulario, frecuencias_df = procesar_corpus(corpus_original)

    estado = {
        "corpus": corpus_preprocesado,
        "preprocesador": Preprocesador(),
        "calculador_tfidf": CalculadorTFIDF(),
        "lista_tokens": lista_tokens,
        "vocabulario": vocabulario,
        "frecuencias_df": frecuencias_df,
        "similitud_coseno": SimilitudCoseno(),
        "matriz_tfidf": None,
        "preprocesamiento_realizado": False,
        "tfidf_realizado": False
    }
    opcion = ""

    while opcion != "0":
        print("\nOpciones:")
        print("1. Ver información del corpus original")
        print("2. Ver información del corpus resultante")
        print("3. Buscar semantico")#pa probar, aun no definida la estructura final de esta.
        print("4. Visualización y análisis exploratorio") #pa probar, aun no definida la estructura final de esta.
        print("0. Salir")

        opcion = input("Seleccione una opción: ")
        if opcion == "0":
            print("Saliendo...")
        elif opcion == "1":
            infoDatasetOriginal(corpus_original)
        elif opcion == "2":
            infoDatasetPreprocesado(corpus_preprocesado, estado)
        elif opcion == "3":
            buscador_semantico(estado)
        elif opcion == "4":
            visualizacion_analisis_exploratorio(estado)
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main()