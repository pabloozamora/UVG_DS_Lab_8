import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import plotly.express as px

# Cargar el modelo y el encoder
regresor_lin = load('regresor_lin.joblib')
encoder_city = load('encoder_city.joblib')

# Título de la aplicación
st.title("Predicción de Costo Total de Renta")

# Descripción de la aplicación
st.write("Por favor, ingrese los detalles de la propiedad para obtener una estimación del costo total de renta.")

# Inputs del usuario
city = st.selectbox("Ciudad", ['São Paulo', 'Porto Alegre', 'Rio de Janeiro', 'Campinas', 'Belo Horizonte'])
area = st.number_input("Área (m²)", min_value=0, value=0)
rooms = st.number_input("Habitaciones", min_value=0, value=0)
bathroom = st.number_input("Baños", min_value=0, value=0)
parking_spaces = st.number_input("Espacios de estacionamiento", min_value=0, value=0)
floor = st.number_input("Piso", min_value=0, value=0)
animal = st.selectbox("¿Se permiten animales?", ['Sí', 'No'])
furniture = st.selectbox("¿Amueblado?", ['Sí', 'No'])
hoa = st.number_input("Cuota HOA (R$)", min_value=0, value=0)
rent_amount = st.number_input("Monto de renta (R$)", min_value=0, value=0)
property_tax = st.number_input("Impuesto de propiedad (R$)", min_value=0, value=0)
fire_insurance = st.number_input("Seguro contra incendios (R$)", min_value=0, value=0)

# Convertir datos de entrada
animal_binary = 1 if animal == 'Sí' else 0
furniture_binary = 1 if furniture == 'Sí' else 0


# Botón para hacer la predicción
if st.button("Predecir Costo Total"):
    # Transformar la entrada usando el encoder
    input_data = encoder_city.transform([[city, area, rooms, bathroom, parking_spaces, floor, animal_binary, furniture_binary, hoa, rent_amount, property_tax, fire_insurance]])
    
    # Obtener la predicción
    prediction = regresor_lin.predict(input_data)
    
    # Mostrar la predicción
    st.write(f"Costo total estimado: R$ {prediction[0][0]:.2f}")
    
# Mostrar la importancia de las características
st.subheader("Importancia de las características")

# Obtener las características codificadas de la ciudad
encoded_city_features = encoder_city.named_transformers_['encoder_city'].get_feature_names_out(['city'])

# Crear la lista completa de nombres de características
all_feature_names = list(encoded_city_features) + [
    "area", "rooms", "bathroom", "parking spaces", "floor", 
    "animal", "furniture", "hoa (R$)", "rent amount (R$)", 
    "property tax (R$)", "fire insurance (R$)"
]

# Obtener coeficientes del modelo y asociarlos a los nombres de las características
feature_importance = np.abs(regresor_lin.coef_.flatten())  # Tomar valor absoluto directamente
importance_df = pd.DataFrame({
    "Característica": all_feature_names,
    "Importancia": feature_importance
})

# Ordenar por importancia (más grande a más pequeña)
importance_df = importance_df.sort_values(by="Importancia", ascending=False)

# Crear una gráfica de pie usando Plotly
fig = px.pie(importance_df, names='Característica', values='Importancia',
             title='Importancia de las Características en el Modelo')

# Mostrar la gráfica en Streamlit
st.plotly_chart(fig)

# Sección para mostrar tendencias de alquiler por ciudad
st.subheader("Tendencias de Alquiler por Ciudad")

rent_data = pd.read_csv('houses_to_rent_v2.csv')

# Agrupar por ciudad y calcular el promedio del costo total
city_trends_mean = rent_data.groupby('city')['total (R$)'].mean().reset_index()

# Crear una gráfica de barras para mostrar el costo promedio por ciudad
trend_fig = px.bar(city_trends_mean, x='city', y='total (R$)', 
                   title='Costo Promedio de Alquiler por Ciudad',
                   labels={'total (R$)': 'Costo Promedio Total (R$)', 'city': 'Ciudad'})

# Mostrar la gráfica en Streamlit
st.plotly_chart(trend_fig)

# Agrupar por ciudad y calcular la moda del costo total
city_trends_mode = rent_data.groupby('city')['total (R$)'].agg(lambda x: x.mode()[0]).reset_index()

# Crear una gráfica de barras para mostrar la moda del costo total por ciudad
trend_fig = px.bar(city_trends_mode, x='city', y='total (R$)', 
                   title='Costo Modal de Alquiler por Ciudad',
                   labels={'total (R$)': 'Costo Modal Total (R$)', 'city': 'Ciudad'})

# Mostrar la gráfica en Streamlit
st.plotly_chart(trend_fig)