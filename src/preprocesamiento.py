class Preprocesador:
    def __init__(self):
        self.stopwords = [
            "the", "and", "of", "to", "in", "that", "it", "is", "was", "he", "for",
            "on", "with", "as", "his", "they", "be", "at", "by", "from", "this",
            "are", "or", "an", "but", "not", "which", "you", "her", "their", "there",
            "were", "all", "him", "them", "had", "have", "has", "my", "me", "your",
            "i", "we", "our", "us", "a"
        ]

    def limpiar_texto(self, texto):
        texto = texto.lower()
        texto_limpio = ""

        for caracter in texto:
            if caracter >= 'a' and caracter <= 'z':
                texto_limpio += caracter
            elif caracter == ' ':
                texto_limpio += caracter
            else:
                texto_limpio += ' '

        return texto_limpio

    def tokenizar(self, texto):
        tokens = texto.split()
        return tokens

    def eliminar_stopwords(self, tokens):
        tokens_filtrados = []

        for palabra in tokens:
            if palabra not in self.stopwords:
                tokens_filtrados.append(palabra)

        return tokens_filtrados

    def preprocesar_texto(self, texto):
        texto_limpio = self.limpiar_texto(texto)
        tokens = self.tokenizar(texto_limpio)
        tokens_filtrados = self.eliminar_stopwords(tokens)

        return texto_limpio, tokens_filtrados

    def construir_vocabulario_y_frecuencias(self, lista_tokens):
        frecuencias = {}

        for tokens in lista_tokens:
            for palabra in tokens:
                if palabra not in frecuencias:
                    frecuencias[palabra] = 1
                else:
                    frecuencias[palabra] += 1

        vocabulario = list(frecuencias.keys())
        vocabulario.sort()

        return vocabulario, frecuencias