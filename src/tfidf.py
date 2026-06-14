import math

class CalculadorTFIDF:
    def __init__(self):
        self.idf = {}
        self.matriz_tfidf = []

    def contar_palabras_documento(self, tokens):
        conteo = {}

        for palabra in tokens:
            if palabra not in conteo:
                conteo[palabra] = 1
            else:
                conteo[palabra] += 1

        return conteo

    def calcular_tf(self, tokens):
        tf = {}
        conteo = self.contar_palabras_documento(tokens)
        total_palabras = len(tokens)

        if total_palabras == 0:
            return tf

        for palabra in conteo:
            tf[palabra] = conteo[palabra] / total_palabras

        return tf

    def calcular_df(self, lista_tokens):
        df = {}

        for tokens in lista_tokens:
            palabras_vistas = []

            for palabra in tokens:
                if palabra not in palabras_vistas:
                    palabras_vistas.append(palabra)

            for palabra in palabras_vistas:
                if palabra not in df:
                    df[palabra] = 1
                else:
                    df[palabra] += 1

        return df

    def calcular_idf(self, lista_tokens):
        df = self.calcular_df(lista_tokens)
        total_documentos = len(lista_tokens)

        self.idf = {}

        for palabra in df:
            self.idf[palabra] = math.log((total_documentos + 1) / (df[palabra] + 1)) + 1

        return self.idf

    def calcular_tfidf_documento(self, tokens):
        tfidf = {}
        tf = self.calcular_tf(tokens)

        for palabra in tf:
            tfidf[palabra] = tf[palabra] * self.idf[palabra]

        return tfidf

    def calcular_tfidf_corpus(self, lista_tokens):
        self.calcular_idf(lista_tokens)

        self.matriz_tfidf = []

        for tokens in lista_tokens:
            tfidf_documento = self.calcular_tfidf_documento(tokens)
            self.matriz_tfidf.append(tfidf_documento)

        return self.matriz_tfidf

    def obtener_top_tfidf(self, tfidf_documento, cantidad):
        items = list(tfidf_documento.items())

        items.sort(key=lambda x: x[1], reverse=True)

        return items[:cantidad]