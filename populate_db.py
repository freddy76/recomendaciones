from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

# Crear sesión
db = SessionLocal()

# Insertar libros de prueba
books_data = [
    {"title": "Falling Up", "author": "Shel Silverstein", "genre": "Poetry"},
    {"title": "The Hobbit", "author": "J.R.R. Tolkien", "genre": "Fantasy"},
    {"title": "1984", "author": "George Orwell", "genre": "Dystopian"},
    {"title": "To Kill a Mockingbird", "author": "Harper Lee", "genre": "Fiction"},
    {"title": "Pride and Prejudice", "author": "Jane Austen", "genre": "Romance"}
]

for book in books_data:
    db.add(models.Book(**book))

db.commit()
db.close()
print("✅ Libros insertados en la base de datos.")
