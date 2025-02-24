import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# 🔹 Cargar dataset de calificaciones
try:
    ratings = pd.read_csv("ratings.csv", sep=";", encoding="utf-8")
    books = pd.read_csv("books.csv", sep=";", encoding="utf-8")  # Cargar libros
    print("✅ Dataset de calificaciones y libros cargado correctamente.")
except Exception as e:
    print(f"❌ Error al leer el CSV: {e}")
    exit()

# 🔹 Verificar columnas
print("\n📝 Columnas en ratings:", ratings.columns.tolist())
print("📝 Columnas en books:", books.columns.tolist())

# 📌 Renombrar columnas para que coincidan con el código
ratings.rename(columns={'User-ID': 'user_id', 'ISBN': 'book_id', 'Rating': 'rating'}, inplace=True)
books.rename(columns={'ISBN': 'book_id', 'Title': 'title'}, inplace=True)

# 📌 Usar solo las columnas necesarias
ratings = ratings[['user_id', 'book_id', 'rating']]
books = books[['book_id', 'title']]

# 📌 Configurar los datos para Surprise
reader = Reader(rating_scale=(1, 10))  # Ajusta según el dataset
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

# 🔹 Dividir datos en entrenamiento y prueba
trainset, testset = train_test_split(data, test_size=0.2)

# 🔹 Entrenar modelo SVD
model = SVD()
model.fit(trainset)

# 🔹 Evaluar el modelo con el conjunto de prueba
predictions = model.test(testset)
print(f"\n📊 RMSE del modelo: {accuracy.rmse(predictions)}")  # Error cuadrático medio

# 📌 Función para recomendar libros con títulos
def recomendar_libros_usuario(user_id, num_recomendaciones=5):
    book_ids = ratings['book_id'].unique()
    
    predicciones = [model.predict(user_id, book_id) for book_id in book_ids]
    predicciones.sort(key=lambda x: x.est, reverse=True)  # Ordenar por calificación predicha
    
    mejores_libros = [p.iid for p in predicciones[:num_recomendaciones]]

    # 📌 Obtener los títulos en lugar de los ISBN
    titulos = books[books['book_id'].isin(mejores_libros)]['title'].tolist()

    return titulos

# 🔹 Probar recomendador con un usuario ejemplo
print("\n📚 Libros recomendados para el usuario 1:")
print(recomendar_libros_usuario(1))
