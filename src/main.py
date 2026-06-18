import pandas as pd
from preprocesamiento import Preprocesador
from tfidf import CalculadorTFIDF
from similitud_coseno import SimilitudCoseno
from procesador_semantico import BuscadorSemantico
from visualizador import VisualizadorCorpus
from clasificador import ClasificadorVersiculos
from generador_ngramas import GeneradorNGramas
from analisis_sentimiento import AnalizadorSentimiento

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

    corpus = corpus[
        ["id", "id_libro", "libro", "testamento", "id_genero", "capitulo", "versiculo", "texto"]
    ]

    print("Corpus base cargado correctamente.")

    return corpus

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

    while True:
        opcion = input("\n¿Desea guardar el corpus preprocesado y las frecuencias en nuevos archivos CSV? (s/n): ").strip().lower()
        if opcion == "s":
            corpus.to_csv("datos/dataset_procesado/corpus_preprocesado.csv", index=False)
            frecuencias_df.to_csv("datos/dataset_procesado/frecuencias_palabras.csv", index=False)
            print("Corpus preprocesado guardado como 'datos/dataset_procesado/corpus_preprocesado.csv'.")
            print("Frecuencias guardadas como 'datos/dataset_procesado/frecuencias_palabras.csv'.")
            break
        elif opcion == "n":
            break
        else:
            print("Ingrese 's' o 'n'.")

def buscador_semantico(estado):

    corpus = estado["corpus"]
    lista_tokens = estado["lista_tokens"]
    preprocesador = estado["preprocesador"]
    calculador_tfidf = estado["calculador_tfidf"]

    matriz_tfidf = calculador_tfidf.calcular_tfidf_corpus(lista_tokens)

    buscador = BuscadorSemantico(corpus, matriz_tfidf, preprocesador, calculador_tfidf)

    consulta = input("\nIngrese su consulta: ")
    if consulta == "":
        print("Ingrese una consulta válida.")
        return
    while True:
        try:
            k = int(input("¿Cuántos resultados desea ver? "))
            if k <= 0:
                print("Ingrese un número mayor a 0.")
                continue
            if k > 20:
                print("Máximo 20 resultados.")
                k = 20
            break
        except ValueError:
            print("Ingrese un número válido.")
    resultados = buscador.buscar(consulta, k)

    print(f"\nResultados para: '{consulta}'")
    for resultado in resultados:
        print("--------------------------------------")
        print("Libro:", resultado["libro"])
        print("Capítulo:", resultado["capitulo"])
        print("Versículo:", resultado["versiculo"])
        print("Similitud:", round(resultado["similitud"], 4))
        print("Texto:", resultado["texto"])

def visualizacion_analisis_exploratorio(estado):
    corpus = estado["corpus"]
    frecuencias_df = estado["frecuencias_df"]
    lista_tokens = estado["lista_tokens"]

    visualizador = VisualizadorCorpus()
    visualizador.generar_visualizaciones_basicas(corpus, frecuencias_df, lista_tokens)

def clasificador_versiculos(estado):
    corpus = estado["corpus"]
    lista_tokens = estado["lista_tokens"]
    etiquetas = corpus["libro"].tolist()

    clasificador = ClasificadorVersiculos()

    tokens_train, tokens_test, y_train, y_test, _ = clasificador.dividir_datos(lista_tokens, etiquetas)

    print(f"Datos de entrenamiento: {len(tokens_train)} versículos")
    print(f"Datos de prueba: {len(tokens_test)} versículos")
    print("\nEntrenando modelo... (puede tardar unos minutos)")

    clasificador.entrenar(tokens_train, y_train)

    print("Modelo entrenado. Prediciendo...")
    y_pred = clasificador.predecir(tokens_test)

    clasificador.evaluar(y_test, list(y_pred))
    clasificador.graficar_confusion(y_test, list(y_pred), "outputs/graficos/")

def generador_texto(estado):

    lista_tokens = estado["lista_tokens"]

    print("\nEntrenando modelos de n-gramas... (puede tardar)")
    generador = GeneradorNGramas()
    generador.entrenar(lista_tokens)
    print("Modelos entrenados.")

    while True:
        palabra = input("\nIngrese una palabra inicial (o 'salir'): ").strip().lower()
        if palabra == "":
            print("Ingrese una palabra válida.")
            continue
        
        if palabra == "salir":
            break

        if palabra not in generador.unigramas:
            print("Palabra no encontrada en el vocabulario, intente otra.")
            continue

        print("\n--- Unigrama ---")
        print(generador.generar_unigrama(palabra))
        print("\n--- Bigrama ---")
        print(generador.generar_bigrama(palabra))
        print("\n--- Trigrama ---")
        print(generador.generar_trigrama(palabra))
        print("\n--- Cuatrigrama ---")
        print(generador.generar_cuatrigrama(palabra))

def analisis_sentimiento(estado):

    corpus = estado["corpus"]

    analizador = AnalizadorSentimiento()

    corpus_sentimiento, sentimiento_libro, sentimiento_capitulo = analizador.analizar_corpus(corpus)

    estado["corpus_sentimiento"] = corpus_sentimiento
    estado["sentimiento_libro"] = sentimiento_libro
    estado["sentimiento_capitulo"] = sentimiento_capitulo

    print("\nProceso finalizando correctamente.")

def main():

    corpus_original = cargar_corpus()
    corpus_preprocesado, lista_tokens, vocabulario, frecuencias_df = procesar_corpus(corpus_original)

    estado = {
        "corpus": corpus_preprocesado,
        "preprocesador": Preprocesador(),
        "calculador_tfidf": CalculadorTFIDF(),
        "similitud_coseno": SimilitudCoseno(),
        "lista_tokens": lista_tokens,
        "vocabulario": vocabulario,
        "frecuencias_df": frecuencias_df,
        "corpus_sentimiento": None,
        "sentimiento_libro": None,
        "sentimiento_capitulo": None
    }
    opcion = ""

    while opcion != "0":
        print("\Menú:")
        print("1. Ver información del corpus original")
        print("2. Ver información del corpus resultante")
        print("3. Buscar semantico")
        print("4. Visualización y análisis exploratorio")
        print("5. Clasificador de versículos")
        print("6. Generador de texto")
        print("7. Análisis de sentimiento")
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
        elif opcion == "5":
            clasificador_versiculos(estado)
        elif opcion == "6":
            generador_texto(estado)
        elif opcion == "7":
            analisis_sentimiento(estado)
        else:
            print("Opción no válida.")

if __name__ == "__main__":
    main()