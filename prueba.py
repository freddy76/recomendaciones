import pandas as pd
import os

# Verificar si el archivo existe
file_path = "books.csv"

if not os.path.exists(file_path):
    print(f"Error: El archivo {file_path} no existe en el directorio.")
else:
    try:
        # Cargar el archivo CSV con el separador correcto (;)
        books = pd.read_csv(file_path, sep=";", encoding="utf-8", engine="python")

        # Mostrar las primeras filas para verificar la carga correcta
        print("Primeras filas del dataset:")
        print(books.head())

        # Mostrar nombres de columnas
        print("\nColumnas detectadas:", books.columns)

        # Verificar si la columna 'description' existe antes de manipularla
        if 'description' in books.columns:
            books['description'] = books['description'].fillna('')  # Reemplazar NaN con texto vacío
        else:
            print("\nLa columna 'description' no existe en el dataset. Se omite este paso.")

    except Exception as e:
        print(f"Error al leer el CSV: {e}")
