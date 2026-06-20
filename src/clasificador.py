from tfidf import CalculadorTFIDF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import random
import matplotlib.pyplot as plt

class ClasificadorVersiculos:
    def __init__(self):
        self.modelo = LogisticRegression(max_iter=1000, C=5, class_weight='balanced')
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
            tf = self.calculador_tfidf.calcular_tf(tokens)
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
    
    def graficar_confusion(self, y_real, y_pred, ruta):
        clases = sorted(set(y_real))
        matriz = confusion_matrix(y_real, y_pred, labels=clases)

        filas = len(matriz)
        columnas = len(matriz[0])
        matriz_norm = []

        for i in range(filas):
            total_fila = 0
            for j in range(columnas):
                total_fila += matriz[i][j]

            fila_norm = []
            for j in range(columnas):
                if total_fila == 0:
                    fila_norm.append(0.0)
                else:
                    fila_norm.append(matriz[i][j] / total_fila)

            matriz_norm.append(fila_norm)

        matriz_norm = np.array(matriz_norm)
        

        plt.figure(figsize=(24, 20))
        
        plt.imshow(matriz_norm, interpolation="nearest", cmap="viridis") 
        plt.grid(False)
        plt.colorbar()
        plt.title("Matriz de confusión (Normalizada)", fontsize=40)
        plt.xticks(range(len(clases)), clases, rotation=90, fontsize=12)
        plt.yticks(range(len(clases)), clases, fontsize=12)
        plt.xlabel("Predicción", fontsize=30)
        plt.ylabel("Real", fontsize=30)
        plt.tight_layout()
        plt.savefig(ruta + "matriz_confusion.png", dpi=300)
        plt.close()