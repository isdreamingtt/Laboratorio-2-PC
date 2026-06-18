import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

plt.style.use("ggplot")


class AnalizadorSentimiento:
    def __init__(self):
        self.analizador = SentimentIntensityAnalyzer()
        self.ruta_graficos = "outputs/graficos/"

    def obtener_etiqueta_sentimiento(self, puntaje):
        if puntaje >= 0.05:
            return "positivo"
        elif puntaje <= -0.05:
            return "negativo"
        else:
            return "neutral"

    def calcular_sentimiento_versiculos(self, corpus):
        corpus_sentimiento = corpus.copy()

        puntajes_negativos = []
        puntajes_neutrales = []
        puntajes_positivos = []
        puntajes_compuestos = []
        etiquetas = []

        for texto in corpus_sentimiento["texto"]:
            resultado = self.analizador.polarity_scores(str(texto))

            puntajes_negativos.append(resultado["neg"])
            puntajes_neutrales.append(resultado["neu"])
            puntajes_positivos.append(resultado["pos"])
            puntajes_compuestos.append(resultado["compound"])

            etiqueta = self.obtener_etiqueta_sentimiento(resultado["compound"])
            etiquetas.append(etiqueta)

        corpus_sentimiento["sentimiento_negativo"] = puntajes_negativos
        corpus_sentimiento["sentimiento_neutral"] = puntajes_neutrales
        corpus_sentimiento["sentimiento_positivo"] = puntajes_positivos
        corpus_sentimiento["sentimiento_compuesto"] = puntajes_compuestos
        corpus_sentimiento["etiqueta_sentimiento"] = etiquetas

        return corpus_sentimiento

    def agregar_sentimiento_por_libro(self, corpus_sentimiento):
        sentimiento_libro = corpus_sentimiento.groupby(
            ["id_libro", "libro", "testamento"]
        )["sentimiento_compuesto"].mean().reset_index()

        sentimiento_libro = sentimiento_libro.rename(
            columns={"sentimiento_compuesto": "sentimiento_promedio"}
        )

        sentimiento_libro = sentimiento_libro.sort_values("id_libro").reset_index(drop=True)

        return sentimiento_libro

    def agregar_sentimiento_por_capitulo(self, corpus_sentimiento):
        sentimiento_capitulo = corpus_sentimiento.groupby(
            ["id_libro", "libro", "testamento", "capitulo"]
        )["sentimiento_compuesto"].mean().reset_index()

        sentimiento_capitulo = sentimiento_capitulo.rename(
            columns={"sentimiento_compuesto": "sentimiento_promedio"}
        )

        sentimiento_capitulo = sentimiento_capitulo.sort_values(
            ["id_libro", "capitulo"]
        ).reset_index(drop=True)

        return sentimiento_capitulo

    def grafico_sentimiento_por_libro(self, sentimiento_libro):
        plt.figure(figsize=(14, 6))
        plt.bar(
            sentimiento_libro["libro"],
            sentimiento_libro["sentimiento_promedio"],
            color="steelblue"
        )

        plt.axhline(0, color="black", linewidth=0.8)
        plt.title("Sentimiento promedio por libro", fontsize=14)
        plt.xlabel("Libro", fontsize=11)
        plt.ylabel("Sentimiento promedio", fontsize=11)
        plt.xticks(rotation=90, fontsize=7)
        plt.tight_layout()
        plt.savefig(self.ruta_graficos + "sentimiento_por_libro.png", dpi=150)
        plt.close()

        print("Gráfico guardado: outputs/graficos/sentimiento_por_libro.png")

    def grafico_sentimiento_por_capitulo(self, sentimiento_capitulo):
        sentimiento_capitulo = sentimiento_capitulo.copy()
        sentimiento_capitulo["indice_capitulo"] = range(1, len(sentimiento_capitulo) + 1)

        plt.figure(figsize=(14, 6))
        plt.plot(
            sentimiento_capitulo["indice_capitulo"],
            sentimiento_capitulo["sentimiento_promedio"],
            color="steelblue",
            linewidth=1
        )

        plt.fill_between(
            sentimiento_capitulo["indice_capitulo"],
            sentimiento_capitulo["sentimiento_promedio"],
            0,
            where=[v >= 0 for v in sentimiento_capitulo["sentimiento_promedio"]],
            color="steelblue",
            alpha=0.3
        )
        plt.fill_between(
            sentimiento_capitulo["indice_capitulo"],
            sentimiento_capitulo["sentimiento_promedio"],
            0,
            where=[v < 0 for v in sentimiento_capitulo["sentimiento_promedio"]],
            color="tomato",
            alpha=0.3
        )

        plt.axhline(0, color="black", linewidth=0.8)
        plt.title("Evolución del sentimiento a lo largo de los capítulos", fontsize=14)
        plt.xlabel("Capítulos en orden bíblico", fontsize=11)
        plt.ylabel("Sentimiento promedio", fontsize=11)
        plt.tight_layout()
        plt.savefig(self.ruta_graficos + "sentimiento_por_capitulo.png", dpi=150)
        plt.close()

        print("Gráfico guardado: outputs/graficos/sentimiento_por_capitulo.png")

    def mostrar_extremos_libros(self, sentimiento_libro):
        libros_positivos = sentimiento_libro.sort_values(
            "sentimiento_promedio",
            ascending=False
        ).head(5)

        libros_negativos = sentimiento_libro.sort_values(
            "sentimiento_promedio",
            ascending=True
        ).head(5)

        print("\n5 libros con sentimiento promedio más positivo:")
        print(libros_positivos[["libro", "testamento", "sentimiento_promedio"]])

        print("\n5 libros con sentimiento promedio más negativo:")
        print(libros_negativos[["libro", "testamento", "sentimiento_promedio"]])

    def mostrar_extremos_capitulos(self, sentimiento_capitulo):
        capitulos_positivos = sentimiento_capitulo.sort_values(
            "sentimiento_promedio",
            ascending=False
        ).head(5)

        capitulos_negativos = sentimiento_capitulo.sort_values(
            "sentimiento_promedio",
            ascending=True
        ).head(5)

        print("\n5 capítulos con sentimiento promedio más positivo:")
        print(capitulos_positivos[["libro", "capitulo", "testamento", "sentimiento_promedio"]])

        print("\n5 capítulos con sentimiento promedio más negativo:")
        print(capitulos_negativos[["libro", "capitulo", "testamento", "sentimiento_promedio"]])

    def analizar_corpus(self, corpus):
        print("\nCalculando sentimiento por versículo...")

        corpus_sentimiento = self.calcular_sentimiento_versiculos(corpus)
        sentimiento_libro = self.agregar_sentimiento_por_libro(corpus_sentimiento)
        sentimiento_capitulo = self.agregar_sentimiento_por_capitulo(corpus_sentimiento)

        print("Generando gráficos de sentimiento...")
        self.grafico_sentimiento_por_libro(sentimiento_libro)
        self.grafico_sentimiento_por_capitulo(sentimiento_capitulo)

        self.mostrar_extremos_libros(sentimiento_libro)
        self.mostrar_extremos_capitulos(sentimiento_capitulo)

        corpus_sentimiento.to_csv("datos/dataset_procesado/corpus_sentimiento.csv", index=False)

        sentimiento_libro.to_csv("datos/dataset_procesado/sentimiento_por_libro.csv",index=False)

        sentimiento_capitulo.to_csv("datos/dataset_procesado/sentimiento_por_capitulo.csv",index=False)

        print("\nArchivos guardados:")
        print("datos/dataset_procesado/corpus_sentimiento.csv")
        print("datos/dataset_procesado/sentimiento_por_libro.csv")
        print("datos/dataset_procesado/sentimiento_por_capitulo.csv")

        return corpus_sentimiento, sentimiento_libro, sentimiento_capitulo