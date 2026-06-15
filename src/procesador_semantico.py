from similitud_coseno import SimilitudCoseno


class BuscadorSemantico:
    def __init__(self, corpus, matriz_tfidf, preprocesador, calculador_tfidf):
        self.corpus = corpus
        self.matriz_tfidf = matriz_tfidf
        self.preprocesador = preprocesador
        self.calculador_tfidf = calculador_tfidf
        self.similitud_coseno = SimilitudCoseno()

    def vectorizar_consulta(self, texto_consulta):
        texto_limpio, tokens = self.preprocesador.preprocesar_texto(texto_consulta)

        tf = self.calculador_tfidf.calcular_tf(tokens)

        vector_consulta = {}

        for palabra in tf:
            if palabra in self.calculador_tfidf.idf:
                vector_consulta[palabra] = tf[palabra] * self.calculador_tfidf.idf[palabra]

        return vector_consulta

    def buscar(self, texto_consulta, k):
        vector_consulta = self.vectorizar_consulta(texto_consulta)

        resultados = []

        for i in range(len(self.matriz_tfidf)):
            vector_versiculo = self.matriz_tfidf[i]

            valor_similitud = self.similitud_coseno.calcular_similitud(
                vector_consulta,
                vector_versiculo
            )

            fila = self.corpus.iloc[i]

            resultado = {
                "libro": fila["libro"],
                "capitulo": fila["capitulo"],
                "versiculo": fila["versiculo"],
                "texto": fila["texto"],
                "similitud": valor_similitud
            }

            resultados.append(resultado)

        resultados.sort(key=lambda x: x["similitud"], reverse=True)

        return resultados[:k]