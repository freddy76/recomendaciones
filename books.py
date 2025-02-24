import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Cargar dataset
try:
    books = pd.read_csv("books.csv", sep=";", encoding="utf-8", engine="python")
    print("✅ Dataset cargado correctamente.")
except Exception as e:
    print(f"❌ Error al leer el CSV: {e}")
    exit()

# Verificar columnas
print("\n📝 Columnas en el dataset:", books.columns.tolist())

# Normalizar títulos y autores para evitar problemas de mayúsculas/minúsculas
books["Title"] = books["Title"].str.strip().str.lower()
books["Author"] = books["Author"].str.strip().str.lower()

# Crear la columna "features" combinando información relevante
books["features"] = books["Title"].fillna("") + " " + books["Author"].fillna("") + " " + books["Publisher"].fillna("")

# Vectorización TF-IDF
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books["features"])

# Crear modelo KNN para buscar similitudes sin calcular la matriz completa
knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(tfidf_matrix)

# Función para recomendar libros con más diversidad
def recomendar_libros(titulo, num_recomendaciones=5):
    titulo = titulo.strip().lower()  # Normalizar título de entrada
    if titulo not in books["Title"].values:
        return f"⚠️ El libro '{titulo}' no está en el dataset."

    # Obtener el índice del libro solicitado
    idx = books[books["Title"] == titulo].index[0]

    # Encontrar los libros más similares con KNN
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recomendaciones*5)  # Buscar más libros

    # Obtener los títulos recomendados, evitando duplicados y autores repetidos
    recomendaciones = []
    autores_vistos = set()  # Para evitar recomendar el mismo autor repetidamente
    for i in indices.flatten():
        recomendado = books.iloc[i]["Title"]
        autor = books.iloc[i]["Author"]

        # Evitar el mismo libro y autores repetidos en la recomendación
        if recomendado != titulo and recomendado not in recomendaciones and autor not in autores_vistos:
            recomendaciones.append(recomendado)
            autores_vistos.add(autor)  # Guardar autor para evitar repeticiones

        # Limitar al número de recomendaciones deseado
        if len(recomendaciones) >= num_recomendaciones:
            break

    return recomendaciones

# Prueba del recomendador
print("\n📚 Libros recomendados:")
print(recomendar_libros("Classical Mythology"))
