import streamlit as st
import os
from dotenv import load_dotenv
import oracledb
import cohere

# Load environment variables
load_dotenv()

# Streamlit app header
st.title("Oracle + Cohere Knowledge Retriever")
st.write("Use this app to query relevant articles from an Oracle DB and generate AI responses.")

# Get user question input
user_question = st.text_input("Enter your question:")

# If the user submits a question, run the query
if user_question:
    cohere_api = os.environ.get("COHERE_API_KEY")
    pw = os.environ.get("PYTHON_PASSWORD")
    dbuser = os.environ.get("DB_USER")
    conn_str = os.environ.get("CONNECT_STRING")

    try:
        # Establish Oracle connection
        connection = oracledb.connect(
            user=dbuser,
            password=pw,
            dsn=conn_str
        )
        st.success("Successfully connected to Oracle Database")
        
        cursor = connection.cursor()

        # Query to fetch relevant articles based on the user's question
        query = """
        SELECT title, abstract 
        FROM CONTENT_ARTICLE
        ORDER BY vector_distance(
            content_article_vector, 
            (vector_embedding(all_minilm_l12_v2 using :query_text as data))
        )
        FETCH FIRST 3 ROWS ONLY
        """
        cursor.execute(query, {'query_text': user_question})
        relevant_articles = cursor.fetchall()

        # Combine titles and abstracts into a context string
        retrieved_context = "\n\n".join([f"{title}: {abstract}" for title, abstract in relevant_articles])

        if retrieved_context.strip():
            # Query Cohere directly
            co = cohere.Client(cohere_api)
            response = co.chat(
                model="command-r-plus",
                message=f"Using this context:\n{retrieved_context}\n\nAnswer the question: {user_question}"
            )
            st.subheader("AI Response:")
            st.write(response.text)
        else:
            st.warning("No relevant articles found for your query.")

        cursor.close()
        connection.close()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Streamlit app footer
st.write("Created by Vraja with ❤️ & Powered by Oracle23ai and Cohere")
