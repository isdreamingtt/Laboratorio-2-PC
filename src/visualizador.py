import matplotlib.pyplot as plt
from tfidf import CalculadorTFIDF
from similitud_coseno import SimilitudCoseno
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np
from matplotlib.patches import Patch

plt.style.use("ggplot")

class VisualizadorCorpus:
    def __init__(self):
        self.ruta_graficos = "outputs/graficos/"

    def grafico_versiculos_por_libro(self, corpus):
        cantidad_por_libro = corpus.groupby("libro").size().sort_values(ascending=True)

        plt.figure(figsize=(12, 10))
        cantidad_por_libro.plot(kind="barh", color="steelblue")
        plt.title("Cantidad de versículos por libro", fontsize=14)
        plt.xlabel("Cantidad de versículos", fontsize=11)
        plt.ylabel("Libro", fontsize=11)
        plt.tight_layout()
        plt.savefig(self.ruta_graficos + "versiculos_por_libro.png", dpi=150)
        plt.close()

        print("Gráfico guardado: outputs/graficos/versiculos_por_libro.png")

    def grafico_palabras_frecuentes(self, frecuencias_df, cantidad):
        top_palabras = frecuencias_df.head(cantidad)

        plt.figure(figsize=(10, 6))
        plt.bar(top_palabras["palabra"], top_palabras["frecuencia"], color="steelblue")
        plt.title("Palabras más frecuentes", fontsize=14)
        plt.xlabel("Palabra", fontsize=11)
        plt.ylabel("Frecuencia", fontsize=11)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.ruta_graficos + "palabras_frecuentes.png", dpi=150)
        plt.close()

        print("Gráfico guardado: outputs/graficos/palabras_frecuentes.png")

    def grafico_longitud_promedio_por_libro(self, corpus):
        longitudes = []

        for tokens in corpus["tokens"]:
            palabras = tokens.split()
            longitudes.append(len(palabras))

        corpus["longitud_tokens"] = longitudes

        longitud_promedio = corpus.groupby("libro")["longitud_tokens"].mean().sort_values(ascending=True)

        plt.figure(figsize=(12, 10))
        longitud_promedio.plot(kind="barh", color="steelblue")
        plt.title("Longitud promedio de versículos por libro", fontsize=14)
        plt.xlabel("Cantidad promedio de palabras", fontsize=11)
        plt.ylabel("Libro", fontsize=11)
        plt.tight_layout()
        plt.savefig(self.ruta_graficos + "longitud_promedio_por_libro.png", dpi=150)
        plt.close()

        print("Gráfico guardado: outputs/graficos/longitud_promedio_por_libro.png")

    def construir_documentos_por_libro(self, corpus):
        libros = []
        lista_tokens_libros = []

        libros_unicos = corpus[["id_libro", "libro"]].drop_duplicates()
        libros_unicos = libros_unicos.sort_values("id_libro")

        for i in range(len(libros_unicos)):
            id_libro = libros_unicos.iloc[i]["id_libro"]
            nombre_libro = libros_unicos.iloc[i]["libro"]

            corpus_libro = corpus[corpus["id_libro"] == id_libro]

            tokens_libro = []

            for tokens_texto in corpus_libro["tokens"]:
                palabras = tokens_texto.split()

                for palabra in palabras:
                    tokens_libro.append(palabra)

            libros.append(nombre_libro)
            lista_tokens_libros.append(tokens_libro)

        return libros, lista_tokens_libros

    def calcular_matriz_similitud_libros(self, lista_tokens_libros):
        calculador_tfidf = CalculadorTFIDF()
        matriz_tfidf_libros = calculador_tfidf.calcular_tfidf_corpus(lista_tokens_libros)

        similitud_coseno = SimilitudCoseno()

        matriz_similitud = []

        for i in range(len(matriz_tfidf_libros)):
            fila_similitud = []

            for j in range(len(matriz_tfidf_libros)):
                similitud = similitud_coseno.calcular_similitud(
                    matriz_tfidf_libros[i],
                    matriz_tfidf_libros[j]
                )

                fila_similitud.append(similitud)

            matriz_similitud.append(fila_similitud)

        return matriz_similitud

    def heatmap_similitud_libros(self, corpus):
        libros, lista_tokens_libros = self.construir_documentos_por_libro(corpus)

        matriz_similitud = self.calcular_matriz_similitud_libros(lista_tokens_libros)

        plt.figure(figsize=(14, 12))
        plt.imshow(matriz_similitud,  interpolation="nearest")
        plt.title("Heatmap de similitud entre libros")
        plt.colorbar(label="Similitud coseno")

        plt.xticks(range(len(libros)), libros, rotation=90, fontsize=6)
        plt.yticks(range(len(libros)), libros, fontsize=6)
        plt.grid(False)
        for i in range(len(libros)):
            plt.hlines(i - 0.5, -0.5, len(libros) - 0.5, colors="white", linewidths=0.3)
            plt.vlines(i - 0.5, -0.5, len(libros) - 0.5, colors="white", linewidths=0.3)

        plt.tight_layout()
        plt.savefig(self.ruta_graficos + "heatmap_similitud_libros.png", dpi=150)
        plt.close()

        print("Heatmap guardado: outputs/graficos/heatmap_similitud_libros.png")

    def generar_visualizaciones_basicas(self, corpus, frecuencias_df,lista_tokens=None):
        self.grafico_versiculos_por_libro(corpus)
        self.grafico_palabras_frecuentes(frecuencias_df, 20)
        self.grafico_longitud_promedio_por_libro(corpus)
        self.heatmap_similitud_libros(corpus)
        if lista_tokens is not None:
            self.grafico_pca_versiculos(corpus, lista_tokens)

    
    def grafico_pca_versiculos(self, corpus, lista_tokens) :

        calculador_tfidf = CalculadorTFIDF()
        matriz_tfidf = calculador_tfidf.calcular_tfidf_corpus(lista_tokens)

        vocabulario = sorted(calculador_tfidf.idf.keys())
        vocab_index = {palabra: i for i, palabra in enumerate(vocabulario)}

        n_docs = len(matriz_tfidf)
        n_vocab = len(vocabulario)
        matriz_densa = np.zeros((n_docs, n_vocab))

        for i, vector in enumerate(matriz_tfidf):
            for palabra, valor in vector.items():
                if palabra in vocab_index:
                    matriz_densa[i][vocab_index[palabra]] = valor

        pca = PCA(n_components=2)
        coordenadas = pca.fit_transform(matriz_densa)

        testamentos = corpus["testamento"].tolist()
        colores = ["steelblue" if t == "OT" else "tomato" for t in testamentos]

        plt.figure(figsize=(12, 8))
        plt.scatter(coordenadas[:, 0], coordenadas[:, 1],
                    c=colores, alpha=0.3, s=5)

        from matplotlib.patches import Patch
        leyenda = [
            Patch(color="steelblue", label="Antiguo Testamento"),
            Patch(color="tomato",    label="Nuevo Testamento")
        ]
        plt.legend(handles=leyenda, fontsize=10)

        varianza = pca.explained_variance_ratio_
        plt.title("PCA de versículos (TF-IDF)", fontsize=14)
        plt.xlabel(f"Componente Principal 1 ({varianza[0]*100:.1f}% varianza)", fontsize=11)
        plt.ylabel(f"Componente Principal 2 ({varianza[1]*100:.1f}% varianza)", fontsize=11)
        plt.tight_layout()
        plt.savefig(self.ruta_graficos + "pca_versiculos.png", dpi=150)
        plt.close()

        print("Gráfico guardado: outputs/graficos/pca_versiculos.png")
        print(f"Varianza explicada: CP1={varianza[0]*100:.1f}%, CP2={varianza[1]*100:.1f}%")