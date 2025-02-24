from fastapi import FastAPI
from typing import List
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from fastapi.middleware.cors import CORSMiddleware
import models
import uvicorn



# Crear instancia de la API
app = FastAPI(title="Recomendador de Libros con Base de Datos")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas las conexiones (puedes restringirlo luego)
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permitir todos los encabezados
)

# Crear las tablas en la base de datos
models.Base.metadata.create_all(bind=engine)

# Dependencia para obtener la sesión de la base de datos
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/books/")
def obtener_libros(db: Session = Depends(get_db)):
    return db.query(models.Book).all()

@app.get("/ratings/")
def obtener_calificaciones(db: Session = Depends(get_db)):
    return db.query(models.Rating).all()
# 🔹 Cargar datasets
try:
    ratings = pd.read_csv("ratings.csv", sep=";", encoding="utf-8")
    books_list = []  # Lista para almacenar los libros temporalmente
    for chunk in pd.read_csv("books.csv", chunksize=1000):
        books_list.append(chunk)

# Unir todos los fragmentos en un solo DataFrame
    books = pd.concat(books_list, ignore_index=True) 
    
    print("✅ Datasets cargados correctamente.")
except Exception as e:
    print(f"❌ Error al leer los archivos CSV: {e}")
    exit()

# 🔹 Renombrar columnas
ratings.rename(columns={'User-ID': 'user_id', 'ISBN': 'book_id', 'Rating': 'rating'}, inplace=True)
books.rename(columns={'ISBN': 'book_id', 'Title': 'title', 'Author': 'author'}, inplace=True)

# 📌 Configurar datos para `Surprise`
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings[['user_id', 'book_id', 'rating']], reader)

# 📌 Dividir datos en entrenamiento y prueba
trainset, testset = train_test_split(data, test_size=0.2)

# 📌 Entrenar modelo SVD
model = SVD()
model.fit(trainset)

# 📌 Filtrado Basado en Contenido (TF-IDF + KNN)
books["features"] = books["title"].fillna("") + " " + books["author"].fillna("")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # Limita características
tfidf_matrix = vectorizer.fit_transform(books["features"])

# 🔹 Modelo KNN para encontrar libros similares
knn_model = NearestNeighbors(metric="cosine", algorithm="auto", n_neighbors=10)
knn_model.fit(tfidf_matrix)

# 📌 Función de Recomendación Basada en Contenido
def recomendar_por_libro(titulo: str, num_recomendaciones: int = 5) -> List[str]:
    """ Recomienda libros similares a un título dado usando TF-IDF + KNN. """
    titulo = titulo.strip().lower()
    
    if titulo not in books["title"].str.lower().values:
        return [f"⚠️ El libro '{titulo}' no está en el dataset."]
    
    # Obtener índice del libro
    idx = books[books["title"].str.lower() == titulo].index[0]
    
    # Encontrar libros similares
    distances, indices = knn_model.kneighbors(tfidf_matrix[idx], n_neighbors=num_recomendaciones+1)
    book_indices = indices.flatten()[1:]  # Excluir el propio libro consultado
    
    return books["title"].iloc[book_indices].tolist()

# 📌 Endpoint para recomendar libros por contenido
@app.get("/recomendar_por_libro/{titulo}", response_model=List[str])
def recomendar_libro_api(titulo: str, num_recomendaciones: int = 5):
    """ API para recomendar libros similares a un libro específico. """
    return recomendar_por_libro(titulo, num_recomendaciones)

# 📌 Ejecutar API con `uvicorn api:app --reload`
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
