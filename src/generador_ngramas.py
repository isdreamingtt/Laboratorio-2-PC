import random
import math

class GeneradorNGramas:
    def __init__(self):
        self.unigramas = {}
        self.bigramas = {}
        self.trigramas = {}
        self.cuatrigramas = {}
        self.total_palabras = 0

    def entrenar(self, lista_tokens):
        for tokens in lista_tokens:
            tokens = ["<START>"] + tokens + ["<END>"]

            for palabra in tokens:
                self.unigramas[palabra] = self.unigramas.get(palabra, 0) + 1
                self.total_palabras += 1

            for i in range(len(tokens) - 1):
                bigrama = (tokens[i], tokens[i+1])
                self.bigramas[bigrama] = self.bigramas.get(bigrama, 0) + 1

            for i in range(len(tokens) - 2):
                trigrama = (tokens[i], tokens[i+1], tokens[i+2])
                self.trigramas[trigrama] = self.trigramas.get(trigrama, 0) + 1

            for i in range(len(tokens) - 3):
                cuatrigrama = (tokens[i], tokens[i+1], tokens[i+2], tokens[i+3])
                self.cuatrigramas[cuatrigrama] = self.cuatrigramas.get(cuatrigrama, 0) + 1

    def _siguiente_palabra_unigrama(self):
        palabras = [p for p in self.unigramas if p not in ("<START>", "<END>")]
        pesos = [self.unigramas[p] for p in palabras]
        return random.choices(palabras, weights=pesos, k=1)[0]

    def _siguiente_palabra_bigrama(self, ultima):
        candidatos = {k[1]: v for k, v in self.bigramas.items() if k[0] == ultima}
        if not candidatos:
            return "<END>"
        return random.choices(list(candidatos.keys()), weights=list(candidatos.values()), k=1)[0]

    def _siguiente_palabra_trigrama(self, penultima, ultima):
        candidatos = {k[2]: v for k, v in self.trigramas.items() if k[0] == penultima and k[1] == ultima}
        if not candidatos:
            return self._siguiente_palabra_bigrama(ultima)
        return random.choices(list(candidatos.keys()), weights=list(candidatos.values()), k=1)[0]

    def _siguiente_palabra_cuatrigrama(self, ante, penultima, ultima):
        candidatos = {k[3]: v for k, v in self.cuatrigramas.items() if k[0] == ante and k[1] == penultima and k[2] == ultima}
        if not candidatos:
            return self._siguiente_palabra_trigrama(penultima, ultima)
        return random.choices(list(candidatos.keys()), weights=list(candidatos.values()), k=1)[0]

    def generar_unigrama(self, palabra_inicial, max_palabras=20):
        resultado = [palabra_inicial]
        for _ in range(max_palabras - 1):
            siguiente = self._siguiente_palabra_unigrama()
            if siguiente == "<END>":
                break
            resultado.append(siguiente)
        return " ".join(resultado)

    def generar_bigrama(self, palabra_inicial, max_palabras=20):
        resultado = [palabra_inicial]
        actual = palabra_inicial
        for _ in range(max_palabras - 1):
            siguiente = self._siguiente_palabra_bigrama(actual)
            if siguiente == "<END>":
                break
            resultado.append(siguiente)
            actual = siguiente
        return " ".join(resultado)

    def generar_trigrama(self, palabra_inicial, max_palabras=20):
        resultado = [palabra_inicial]
        sig = self._siguiente_palabra_bigrama(palabra_inicial)
        if sig == "<END>":
            sig = self._siguiente_palabra_unigrama()
        resultado.append(sig)

        for _ in range(max_palabras - 2):
            siguiente = self._siguiente_palabra_trigrama(resultado[-2], resultado[-1])
            if siguiente == "<END>":
                break
            resultado.append(siguiente)
        return " ".join(resultado)

    def generar_cuatrigrama(self, palabra_inicial, max_palabras=20):
        resultado = [palabra_inicial]

        sig = self._siguiente_palabra_bigrama(palabra_inicial)
        if sig == "<END>":
            sig = self._siguiente_palabra_unigrama()
        resultado.append(sig)

        sig = self._siguiente_palabra_trigrama(resultado[-2], resultado[-1])
        if sig == "<END>":
            sig = self._siguiente_palabra_unigrama()
        resultado.append(sig)

        for _ in range(max_palabras - 3):
            siguiente = self._siguiente_palabra_cuatrigrama(resultado[-3], resultado[-2], resultado[-1])
            if siguiente == "<END>":
                break
            resultado.append(siguiente)
        return " ".join(resultado)