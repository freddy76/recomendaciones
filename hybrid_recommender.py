import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import RandomizedSearchCV
from surprise import accuracy
from surprise.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# 🔹 Cargar datasets
try:
    ratings = pd.read_csv("ratings.csv", sep=";", encoding="utf-8")
    books = pd.read_csv("books.csv", sep=";", encoding="utf-8")
    print("✅ Datasets cargados correctamente.")
except Exception as e:
    print(f"❌ Error al leer los archivos CSV: {e}")
    exit()

# 🔹 Renombrar columnas para que coincidan en ambos sistemas
ratings.rename(columns={'User-ID': 'user_id', 'ISBN': 'book_id', 'Rating': 'rating'}, inplace=True)
books.rename(columns={'ISBN': 'book_id', 'Title': 'title', 'Author': 'author'}, inplace=True)

# 📌 Usar solo las columnas necesarias
ratings = ratings[['user_id', 'book_id', 'rating']]
books = books[['book_id', 'title', 'author']]

# 📌 Configurar los datos para `Surprise` (Filtrado Colaborativo)
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

# 🔹 Ajustar hiperparámetros de SVD usando Random
print("\n🔍 Optimizando SVD con RandomizedSearchCV...")
param_grid = {
    'n_factors': [50, 100, 150],  
    'lr_all': [0.002, 0.005, 0.01],  
    'reg_all': [0.02, 0.05, 0.1]  
}

rs = RandomizedSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_iter=5, n_jobs=-1)
rs.fit(data)

# 📌 Obtener el mejor modelo optimizado
best_params = rs.best_params['rmse']
print(f"✅ Mejores parámetros encontrados: {best_params}")
model = SVD(**best_params)

# 🔹 Dividir datos en entrenamiento y prueba correctamente
trainset, testset = train_test_split(data, test_size=0.2)  # ✅ SE APLICA SOBRE `data`, NO `trainset`

# 🔹 Entrenar modelo optimizado
model = SVD(**best_params)
model.fit(trainset)  # ✅ Entrenar con `trainset`

# 🔹 Evaluar el modelo con RMSE optimizado
predictions = model.test(testset)
print(f"\n📊 RMSE del modelo optimizado: {accuracy.rmse(predictions)}")

# 📌 Configurar Filtrado Basado en Contenido
books["features"] = books["title"].fillna("") + " " + books["author"].fillna("")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # Limita características para mejor rendimiento
tfidf_matrix = vectorizer.fit_transform(books["features"])

# 🔹 Modelo KNN optimizado (menor carga computacional)
knn_model = NearestNeighbors(metric="cosine", algorithm="auto", n_neighbors=10)
knn_model.fit(tfidf_matrix)

# 📌 Función de Filtrado Colaborativo (SVD)
def recomendar_por_usuario(user_id, num_recomendaciones=5):
    """ Recomienda libros basados en el historial de calificaciones del usuario usando SVD. """
    book_ids = ratings['book_id'].unique()
    predicciones = [model.predict(user_id, book_id) for book_id in book_ids]
    predicciones.sort(key=lambda x: x.est, reverse=True)
    mejores_libros = [p.iid for p in predicciones[:num_recomendaciones]]
    return books[books['book_id'].isin(mejores_libros)]['title'].tolist()

# 📌 Función de Filtrado Basado en Contenido
def recomendar_por_libro(titulo, num_recomendaciones=5):
    """ Recomienda libros similares a un libro específico usando TF-IDF + KNN. """
    titulo = titulo.strip().lower()
    if titulo not in books["title"].str.lower().values:
        return f"⚠️ El libro '{titulo}' no está en el dataset."

    idx = books[books["title"].str.lower() == titulo].index[0]
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recomendaciones+1)
    book_indices = indices.flatten()[1:]
    return books["title"].iloc[book_indices].tolist()

# 📌 Función de Recomendación Híbrida Mejorada
def recomendar_hibrido(user_id, libro_titulo=None, num_recomendaciones=5):
    """ Combina Filtrado Colaborativo y Basado en Contenido para recomendar libros con un mejor balance. """
    recomendaciones_usuario = recomendar_por_usuario(user_id, num_recomendaciones)
    recomendaciones_libro = recomendar_por_libro(libro_titulo, num_recomendaciones) if libro_titulo else []

    # 📌 Balancear recomendaciones: Si el usuario tiene muchas calificaciones, darle más peso a colaborativo
    peso_colaborativo = min(len(ratings[ratings['user_id'] == user_id]) / 10, 1)  # Máximo peso 1.0
    peso_contenido = 1 - peso_colaborativo  # Complementario

    # 📌 Fusionar recomendaciones con pesos dinámicos
    recomendaciones_finales = (recomendaciones_usuario[:int(num_recomendaciones * peso_colaborativo)] +
                               recomendaciones_libro[:int(num_recomendaciones * peso_contenido)])

    return list(dict.fromkeys(recomendaciones_finales))  # Evitar duplicados

# 🔹 Probar recomendador híbrido optimizado
print("\n📚 Recomendaciones combinadas optimizadas para el usuario 1 y el libro 'Falling Up':")
print(recomendar_hibrido(user_id=1, libro_titulo="Falling Up"))
