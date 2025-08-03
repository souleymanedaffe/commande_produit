import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from streamlit_folium import st_folium

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


Region = {
    'Central': [41.8781, -87.6298],  # Chicago
    'East': [40.7128, -74.0060],     # New York
    'West': [34.0522, -118.2437],    # Los Angeles
    'South': [29.7604, -95.3698]     # Houston
}


# Charger le dataset pour encoder les valeurs
@st.cache_data
def load_data():
    df = pd.read_excel("Superstore.xlsx", sheet_name="Superstore")

    # Conversion des dates
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    features = ['Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount', 'Region', 'Segment', 'Ship Mode']
    df_model = df[features + ['Profit']].copy()
    df_model['Rentable'] = (df_model['Profit'] > 0).astype(int)
    df_model.drop(columns=['Profit'], inplace=True)

    label_encoders = {}
    for col in df_model.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col])
        label_encoders[col] = le

    X = df_model.drop(columns=['Rentable'])
    y = df_model['Rentable']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    return df, model, label_encoders

# Charger les données, modèle et encodeurs
df, model, label_encoders = load_data()

st.title(" Prédicteur de rentabilité de commande - Superstore")
st.markdown("Entrez les caractéristiques de la commande ci-dessous :")

# Inputs utilisateur
category = st.selectbox("Catégorie", label_encoders['Category'].classes_)
sub_category = st.selectbox("Sous-catégorie", label_encoders['Sub-Category'].classes_)
ship_mode = st.selectbox("Mode de livraison", label_encoders['Ship Mode'].classes_)
segment = st.selectbox("Segment client", label_encoders['Segment'].classes_)
region = st.selectbox("Région", label_encoders['Region'].classes_)
sales = st.number_input("Montant des ventes (€)", min_value=0.0, step=10.0)
quantity = st.number_input("Quantité commandée", min_value=1, step=1)
discount = st.slider("Remise appliquée (%)", min_value=0.0, max_value=1.0, step=0.05)



# Prédiction
if st.button("Prédire la rentabilité"):
    input_data = pd.DataFrame({
        'Category': [label_encoders['Category'].transform([category])[0]],
        'Sub-Category': [label_encoders['Sub-Category'].transform([sub_category])[0]],
        'Sales': [sales],
        'Quantity': [quantity],
        'Discount': [discount],
        'Region': [label_encoders['Region'].transform([region])[0]],
        'Segment': [label_encoders['Segment'].transform([segment])[0]],
        'Ship Mode': [label_encoders['Ship Mode'].transform([ship_mode])[0]]
    })

    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][prediction]

    # Affichage du résultat
    if prediction == 1:
        st.success(f"✅ La commande est probablement **rentable** avec une confiance de {proba:.2%}.")
    else:
        st.error(f"❌ La commande risque de **ne pas être rentable** avec une confiance de {proba:.2%}.")

    explanation = []
    if discount > 0.3:
        explanation.append("La remise est élevée, ce qui peut réduire le profit.")
    if sales < 50:
        explanation.append("Le montant des ventes est faible.")
    if quantity > 5:
        explanation.append("Une grande quantité commandée augmente les coûts.")
    if not explanation:
        explanation.append("Les facteurs semblent équilibrés en faveur d'une commande rentable.")

    st.markdown("### 🔎 Raisons possibles :")
    for reason in explanation:
        st.markdown(f"- {reason}")

    # Afficher les features importantes
    st.subheader("🔍 Variables influentes dans le modèle")
    importance_df = pd.DataFrame({
        'Feature': input_data.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.dataframe(importance_df)

# Graphiques d'analyse
st.header("📊 Analyses Visuelles des Données")

# Rentabilité par catégorie
st.subheader("Rentabilité moyenne par catégorie")
category_profit = df.groupby('Category')['Profit'].mean().sort_values()
fig1, ax1 = plt.subplots()
category_profit.plot(kind='barh', ax=ax1)
ax1.set_title("Profit moyen par catégorie")
ax1.set_xlabel("Profit moyen")
st.pyplot(fig1)


# Heatmap de corrélation
st.subheader("Corrélation entre les variables numériques")
numeric_df = df[['Sales', 'Quantity', 'Discount', 'Profit']]
fig3, ax3 = plt.subplots()
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax3)
ax3.set_title("Matrice de corrélation")
st.pyplot(fig3)

# Ventes mensuelles
st.subheader("Ventes au fil du temps")
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
fig4, ax4 = plt.subplots()
monthly_sales.plot(ax=ax4)
ax4.set_title("Ventes mensuelles")
ax4.set_ylabel("Montant des ventes (€)")
ax4.set_xlabel("Date")
st.pyplot(fig4)

# Ventes par région
st.subheader("Ventes par région")
fig5, ax5 = plt.subplots()
df.groupby('Region')['Sales'].sum().plot(kind='bar', ax=ax5)
ax5.set_title("Ventes par région")
ax5.set_ylabel("Montant des ventes (€)")
st.pyplot(fig5)

# Répartition des ventes par segment
st.subheader("Répartition des ventes par segment de client")
fig6, ax6 = plt.subplots()
df.groupby('Segment')['Sales'].sum().plot(kind='pie', autopct='%1.1f%%', ax=ax6)
ax6.set_ylabel('')
ax6.set_title("Part des ventes par segment")
st.pyplot(fig6)



@st.cache_data
def get_city_coordinates(city_state_df):
    geolocator = Nominatim(user_agent="superstore_map")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    coords = []

    for _, row in city_state_df.iterrows():
        location = geocode(f"{row['City']}, {row['State']}, USA")
        if location:
            coords.append((row['City'], row['State'], location.latitude, location.longitude))
        else:
            coords.append((row['City'], row['State'], None, None))

    coords_df = pd.DataFrame(coords, columns=['City', 'State', 'Latitude', 'Longitude'])
    return coords_df.dropna()


# Carte interactive des clients par région
st.subheader("📍 Répartition géographique des clients par région")
region_clients = df.groupby('Region')['Customer ID'].nunique().reset_index()
region_clients = region_clients.rename(columns={'Customer ID': 'Nombre de clients'})

map_center = [39.8283, -98.5795]  # Centre des États-Unis
m = folium.Map(location=map_center, zoom_start=4)

for _, row in region_clients.iterrows():
    region_name = row['Region']
    count = row['Nombre de clients']
    coords = Region.get(region_name)
    if coords:
        folium.Marker(
            location=coords,
            popup=f"<b>Région:</b> {region_name}<br><b>Nombre de clients:</b> {count}",
            tooltip=f"{region_name}: {count} clients"
        ).add_to(m)
    else:
        st.warning(f"Coordonnées manquantes pour la région: {region_name}")

st_folium(m, width=700, height=500)
