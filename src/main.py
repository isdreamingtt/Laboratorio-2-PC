import pandas as pd


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


    print("\nPrimeras filas del corpus:")
    
    print(corpus.head())

    print("\nCantidad total de versículos:")
    print(len(corpus))

    print("\nCantidad total de libros:")
    print(corpus["libro"].nunique())

    print("\nCantidad por testamento:")
    print(corpus["testamento"].value_counts())

    corpus.to_csv("datos/dataset_procesado/corpus_base.csv", index=False)

    print("\nArchivo guardado en datos/dataset_procesado/corpus_base.csv")


if __name__ == "__main__":
    main()