from tfidf import CalculadorTFIDF
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import random

class ClasificadorVersiculos:
    def __init__(self):
        self.modelo = LogisticRegression(max_iter=1000, C=5)
        self.calculador_tfidf = CalculadorTFIDF()
        self.vocabulario = []
        self.vocab_index = {}
        self.libros_clases = []

    def construir_matriz_densa(self, matriz_tfidf):
        n_docs = len(matriz_tfidf)
        n_vocab = len(self.vocabulario)
        matriz = np.zeros((n_docs, n_vocab))

        for i, vector in enumerate(matriz_tfidf):
            for palabra, valor in vector.items():
                if palabra in self.vocab_index:
                    matriz[i][self.vocab_index[palabra]] = valor

        return matriz

    def entrenar(self, lista_tokens, etiquetas):
        matriz_tfidf = self.calculador_tfidf.calcular_tfidf_corpus(lista_tokens)

        self.vocabulario = sorted(self.calculador_tfidf.idf.keys())
        self.vocab_index = {p: i for i, p in enumerate(self.vocabulario)}

        X = self.construir_matriz_densa(matriz_tfidf)
        
        X = X - X.min()

        self.libros_clases = etiquetas
        self.modelo.fit(X, etiquetas)

        return X

    def dividir_datos(self, lista_tokens, etiquetas, proporcion_test=0.2, semilla=42):
        random.seed(semilla)
        indices = list(range(len(lista_tokens)))
        random.shuffle(indices)

        corte = int(len(indices) * (1 - proporcion_test))
        idx_train = indices[:corte]
        idx_test = indices[corte:]

        tokens_train = [lista_tokens[i] for i in idx_train]
        tokens_test  = [lista_tokens[i] for i in idx_test]
        y_train = [etiquetas[i] for i in idx_train]
        y_test  = [etiquetas[i] for i in idx_test]

        return tokens_train, tokens_test, y_train, y_test, idx_test

    def predecir(self, tokens_test):
        matriz_tfidf_test = []

        for tokens in tokens_test:
            from tfidf import CalculadorTFIDF as C
            tf_calc = C()
            tf = tf_calc.calcular_tf(tokens)
            vector = {}
            for palabra in tf:
                if palabra in self.calculador_tfidf.idf:
                    vector[palabra] = tf[palabra] * self.calculador_tfidf.idf[palabra]
            matriz_tfidf_test.append(vector)

        X_test = self.construir_matriz_densa(matriz_tfidf_test)
        X_test = X_test - X_test.min()

        return self.modelo.predict(X_test)

    def evaluar(self, y_real, y_pred):
        acc = accuracy_score(y_real, y_pred)
        print(f"\nAccuracy: {acc*100:.2f}%")

        clases = sorted(set(y_real))
        matriz = confusion_matrix(y_real, y_pred, labels=clases)

        print("\nMatriz de confusión (primeros 10 libros):")
        clases_cortas = clases[:10]
        sub_real = [y for y in y_real if y in clases_cortas]
        sub_pred = [y_pred[i] for i, y in enumerate(y_real) if y in clases_cortas]
        sub_matriz = confusion_matrix(sub_real, sub_pred, labels=clases_cortas)

        print("Libro".ljust(25) + "".join([str(c)[:4].ljust(6) for c in clases_cortas]))
        for i, fila in enumerate(sub_matriz):
            print(clases_cortas[i].ljust(25) + "".join([str(v).ljust(6) for v in fila]))

        return acc, matriz