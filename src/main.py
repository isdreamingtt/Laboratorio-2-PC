import pandas as pd
from preprocesamiento import Preprocesador
from tfidf import CalculadorTFIDF

def main():
    print("Iniciando lectura del dataset...")
    print("Iniciando procesamiento del corpus...")

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

    print("\nPrimeras 5 filas del corpus base:")
    print(corpus.head())

    print("\nCantidad total de versículos:")
    print(len(corpus))

    print("\nCantidad total de libros:")
    print(corpus["libro"].nunique())

    print("\nCantidad por testamento:")
    print(corpus["testamento"].value_counts())

    preprocesador = Preprocesador()

    textos_limpios = []
    tokens_procesados = []
    lista_tokens = []

    for texto in corpus["texto"]:
        texto_limpio, tokens = preprocesador.preprocesar_texto(texto)

        textos_limpios.append(texto_limpio)
        tokens_procesados.append(" ".join(tokens))
        lista_tokens.append(tokens)

    corpus["texto_limpio"] = textos_limpios
    corpus["tokens"] = tokens_procesados

    print("\nPrimeras 5 filas del corpus preprocesado:")
    print(corpus.head())

    vocabulario, frecuencias = preprocesador.construir_vocabulario_y_frecuencias(lista_tokens)

    print("\nCantidad de palabras únicas en el vocabulario:")
    print(len(vocabulario))

    frecuencias_df = pd.DataFrame(list(frecuencias.items()), columns=["palabra", "frecuencia"])
    frecuencias_df = frecuencias_df.sort_values("frecuencia", ascending=False).reset_index(drop=True)

    print("\n20 palabras más frecuentes:")
    print(frecuencias_df.head(20))

    print("\nCalculando TF-IDF -> prueba para ver si funciona")

    calculador_tfidf = CalculadorTFIDF()
    
    matriz_tfidf = calculador_tfidf.calcular_tfidf_corpus(lista_tokens)

    print("\nCantidad de documentos vectorizados:", len(matriz_tfidf))
    print("\nCantidad de palabras con IDF:", len(calculador_tfidf.idf))

    print("\nTop 10 palabras TF-IDF del primer versículo:")
    top_primer_versiculo = calculador_tfidf.obtener_top_tfidf(matriz_tfidf[0], 10)

    for palabra, valor in top_primer_versiculo:
        print(palabra, ":", valor)

    #Calculo de tf-idf para el primer versículo, solo es una prueba para verificar que el proceso se realizó correctamente.
    

    corpus.to_csv("datos/dataset_procesado/corpus_preprocesado.csv", index=False)
    frecuencias_df.to_csv("datos/dataset_procesado/frecuencias_palabras.csv", index=False)

    print("\nArchivo guardado en:")
    print("datos/dataset_procesado/corpus_preprocesado.csv")

    print("\nFrecuencias guardadas en:")
    print("datos/dataset_procesado/frecuencias_palabras.csv")

if __name__ == "__main__":
    main()