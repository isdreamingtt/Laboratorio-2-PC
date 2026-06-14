class SimilitudCoseno:
    def __init__(self):
        pass

    def producto_punto(self, vector_a, vector_b):
        resultado = 0

        for palabra in vector_a:
            if palabra in vector_b:
                resultado += vector_a[palabra] * vector_b[palabra]

        return resultado

    def magnitud(self, vector):
        suma = 0

        for palabra in vector:
            suma += vector[palabra] ** 2

        return suma ** 0.5

    def calcular_similitud(self, vector_a, vector_b):
        producto = self.producto_punto(vector_a, vector_b)
        magnitud_a = self.magnitud(vector_a)
        magnitud_b = self.magnitud(vector_b)

        if magnitud_a == 0 or magnitud_b == 0:
            return 0

        similitud = producto / (magnitud_a * magnitud_b)

        return similitud