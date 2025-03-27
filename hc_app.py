import streamlit as st
import cohere
import os
import pandas as pd
import plotly.express as px
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random  # For outcome simulation
import json  # For serializing structured context

# Load environment variables
load_dotenv('~/app/.env')
cohere_api = os.environ.get("COHERE_API_KEY")
pw = os.environ.get("PYTHON_PASSWORD")
dbuser = os.environ.get("DB_USER")
conn_str = os.environ.get("CONNECT_STRING")
hostname = os.environ.get("HOSTNAME")
service_name = os.environ.get("SERVICE_NAME")

# Function to generate doctor notes
def generate_doctor_notes(similar_cases):
    co = cohere.Client(cohere_api)
    retrieved_context = "\n\n".join([f"Patient {c[1]}: {c[3]}, {c[4]}. Notes: {c[6]}" for c in similar_cases])
    response = co.chat(model="command-r-plus", message=f"Using this context:\n{retrieved_context}\n\nGenerate doctor notes and treatment suggestions.")
    return response.text

# Database Connection 
def get_db_connection():
    conn_str = f"oracle+oracledb://{dbuser}:{pw}@{hostname}:1521/?service_name={service_name}".format(dbuser=dbuser, pw=pw, hostname=hostname, service_name=service_name)
    engine = create_engine(conn_str)
    return engine

# Fetch data from the database
def get_all_patient_data():
    query = "SELECT PATIENT_ID, NAME, AGE, DIAGNOSIS, SYMPTOMS, LOCATION, PATIENT_VECTOR FROM OMLUSER.HEALTHCARE_DATA_VECTOR"
    try:
        engine = get_db_connection()
        df = pd.read_sql(query, engine)
        if df.empty:
            raise Exception("No data found in the database.")
        return df
    except Exception as e:
        st.error(f"Error fetching patient data: {e}")
    return pd.DataFrame()

# Function to search for similar patients based on the query text
def search_patients(query_text):
    query = """SELECT PATIENT_ID, NAME, AGE, DIAGNOSIS, SYMPTOMS, MEDICATIONS, DOCTOR_NOTES, LAB_RESULTS, COSINE_DISTANCE(PATIENT_VECTOR, (SELECT VECTOR_EMBEDDING(model USING :query_text AS DATA) FROM DUAL)) AS similarity_score 
               FROM OMLUSER.HEALTHCARE_DATA_VECTOR ORDER BY similarity_score ASC FETCH FIRST 5 ROWS ONLY"""
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(text(query), {'query_text': query_text})
        results = result.fetchall()
    return results

# Function to get disease clusters
def get_disease_clusters():
    query = """SELECT DIAGNOSIS, COUNT(1) AS patient_count FROM OMLUSER.HEALTHCARE_DATA_VECTOR GROUP BY DIAGNOSIS ORDER BY patient_count DESC"""
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(text(query))
        disease_clusters = result.fetchall()
    return pd.DataFrame(disease_clusters, columns=['DIAGNOSIS', 'PATIENT_COUNT'])

# Function to get symptoms clusters
def get_symptom_clusters():
    query = """SELECT SYMPTOMS, COUNT(1) AS symptom_count FROM OMLUSER.HEALTHCARE_DATA_VECTOR GROUP BY SYMPTOMS ORDER BY symptom_count DESC"""
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(text(query))
        symptoms_clusters = result.fetchall()
    return pd.DataFrame(symptoms_clusters, columns=['SYMPTOMS', 'SYMPTOM_COUNT'])

# Function to get patient stats (total patients, critical cases)
def get_patient_stats():
    query = """SELECT COUNT(1) AS total_patients, COUNT(CASE WHEN DIAGNOSIS LIKE '%Critical%' THEN 1 END) AS critical_cases 
               FROM OMLUSER.HEALTHCARE_DATA_VECTOR"""
    engine = get_db_connection()
    with engine.connect() as conn:
        result = conn.execute(text(query))
        stats = result.fetchone()
    return stats

# Function to generate the word cloud image
def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Function to simulate outcome prediction
def simulate_outcome_prediction(treatment):
    success_rate = random.random()
    return "Success" if success_rate > 0.3 else "Failure"

# Streamlit UI
st.title("AI-Powered Healthcare Analytics")

# Dashboard View
st.header("Key Healthcare Metrics")
total_patients, critical_cases = get_patient_stats()
st.metric(label="Total Patients", value=total_patients)
st.metric(label="Critical Cases", value=critical_cases)

# Disease Trends - Disease Clusters (Using the updated function)
st.header("Most Common Diseases")
disease_clusters = get_disease_clusters()

if not disease_clusters.empty:
    diagnoses_text = " ".join(disease_clusters['DIAGNOSIS'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(diagnoses_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    st.warning("No data available for disease diagnoses.")

# Symptoms by Diagnosis (Word Cloud for each Diagnosis)
st.header("Most Common Symptoms by Diagnosis")
symptom_clusters = get_symptom_clusters()
if not symptom_clusters.empty:
    symptoms_text = " ".join(symptom_clusters['SYMPTOMS'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(symptoms_text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
else:
    st.warning("No data available for symptoms.")

# AI-Driven Patient Search
st.header("AI-Driven Patient Search")
query_text = st.text_input("Enter symptoms, medications, or conditions to find similar patients:")
if query_text:
    results = search_patients(query_text)
    if results:
        total_cases = len(results)
        st.subheader(f"Total similar patients found: {total_cases}")
        patient_data = get_all_patient_data()
        similar_patients_df = pd.DataFrame(results, columns=["PATIENT_ID", "NAME", "AGE", "DIAGNOSIS", "SYMPTOMS", "MEDICATIONS", "DOCTOR_NOTES", "LAB_RESULTS", "Similarity Score"])

        # Diagnosis Distribution (Bar Chart)
        diagnosis_counts = similar_patients_df['DIAGNOSIS'].value_counts()
        fig_diagnosis = px.bar(diagnosis_counts, title="Diagnosis Distribution of Similar Patients", labels={'value': 'Count'}, height=400)
        st.plotly_chart(fig_diagnosis)

        # Age Distribution by Diagnosis (Boxplot)
        st.header("Age Distribution by Diagnosis")
        fig_age_diagnosis = px.box(similar_patients_df, x='DIAGNOSIS', y='AGE', title="Age Distribution by Diagnosis")
        st.plotly_chart(fig_age_diagnosis)

        # Results 
        st.header("Doctor Notes and Recommendations")
        ai_notes = generate_doctor_notes(results)
        st.write(ai_notes)

        # Symptoms by Diagnosis (Word Cloud for each Diagnosis)
        st.header("Most Common Symptoms by Diagnosis")
        for diagnosis in similar_patients_df['DIAGNOSIS'].unique():
            symptoms_in_diagnosis = similar_patients_df[similar_patients_df['DIAGNOSIS'] == diagnosis]['SYMPTOMS'].tolist()
            symptoms_text = ' '.join(symptoms_in_diagnosis)
            st.subheader(f"Symptoms for {diagnosis}")
            st.image(create_wordcloud(symptoms_text).to_array(), use_container_width=True)

        # Personalized Insights
        st.subheader("Personalized Insights")
        st.write("Based on similar patient cases, here are some recommendations:")
        st.write("- Consider starting treatment with a specific medication that was effective for similar patients.")
        st.write("- Monitor vital signs closely as there may be a risk for complications seen in other cases.")
    else:
        st.warning("No similar patients found.")
st.success("Real-time data updates via Oracle GoldenGate!")
