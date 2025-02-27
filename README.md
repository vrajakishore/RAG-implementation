# Retrieving and Generating Embeddings with Oracle Database and all-MiniLM-L12-v2


## What is a Vector Search?

A Vector search is a technique to find similar data points based on the their numerical representation, known as embeddings in a multi-dimensional space. Instead of performing keyword based search, vector search relies on semantic similarity.

For detailed explanation and capabilities : Oracle AI Vector Search

## What is RAG (Retrieval-Augmented Generation)?

RAG, or Retrieval-Augmented Generation, is a technique that enhances LLM (Large Language Model) responses by dynamically retrieving relevant data from an external source —like a vector DB like Oracle 23ai — before generating an answer. Instead of relying solely on pre-trained knowledge, RAG first performs a vector search to find the most relevant documents and then uses that context to generate a more accurate and informed response. This makes RAG particularly useful for applications like chatbots, search engines, and AI-driven assistants that require up-to-date or domain-specific knowledge.

Here’s a basic representation:

[Knowledge Base] — ->[Retrieval] — → [LLM] — —> [Generation] — → [Output]

## Steps for the implementation.

## 📥 Download the Dataset

```bash
curl -L -o train.csv "https://huggingface.co/datasets/prithivMLmods/Content-Articles/resolve/main/datasets/train.csv"
```

## 📌 Loading the Augmented all-MiniLM-L12-v2 Model into Oracle Database

## 1️⃣ Unzip the ONNX Model

```bash
unzip all-MiniLM-L12-v2_augmented.zip
```

## 2️⃣ Log into the Database as SYSDBA

```sql
sqlplus / as sysdba
```

```sql
ALTER SESSION SET CONTAINER=<pdb>;
```

## 3️⃣ Grant Necessary Privileges and Define Data Dump Directory

```sql
GRANT DB_DEVELOPER_ROLE, CREATE MINING MODEL TO <dbuser>;
CREATE OR REPLACE DIRECTORY DM_DUMP AS '<directory_path>';
GRANT READ ON DIRECTORY DM_DUMP TO <dbuser>;
GRANT WRITE ON DIRECTORY DM_DUMP TO <dbuser>;
EXIT;
```

## 4️⃣ Log in as OMLUSER

```sql
sqlplus <dbuser>/<password>@<pdb>
```

## 5️⃣ Load the ONNX Model

```sql
EXEC DBMS_VECTOR.DROP_ONNX_MODEL(model_name => 'ALL_MINILM_L12_V2', force => true);

BEGIN
   DBMS_VECTOR.LOAD_ONNX_MODEL(
        directory => 'DM_DUMP',
        file_name => 'all_MiniLM_L12_v2.onnx',
        model_name => 'ALL_MINILM_L12_V2',
        metadata => JSON('{"function" : "embedding", "embeddingOutput" : "embedding", "input": {"input": ["DATA"]}}'));
END;
/
```

## 6️⃣ Validate the Imported Model

```sql
SELECT model_name, algorithm, mining_function
FROM user_mining_models
WHERE model_name='ALL_MINILM_L12_V2';
```

## 7️⃣ Generate Embedding Vectors

```sql
SELECT VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING 'The quick brown fox jumped' AS DATA) AS embedding;
```

## 8️⃣ Alternative Method to Import ONNX Models

```sql
EXEC DBMS_VECTOR.DROP_ONNX_MODEL(model_name => 'ALL_MINILM_L12_V2', force => true);

DECLARE
    m_blob BLOB DEFAULT EMPTY_BLOB();
    m_src_loc BFILE;
BEGIN
    DBMS_LOB.CREATETEMPORARY(m_blob, FALSE);
    m_src_loc := BFILENAME('DM_DUMP', 'all_MiniLM_L12_v2.onnx');
    DBMS_LOB.FILEOPEN(m_src_loc, DBMS_LOB.FILE_READONLY);
    DBMS_LOB.LOADFROMFILE(m_blob, m_src_loc, DBMS_LOB.GETLENGTH(m_src_loc));
    DBMS_LOB.CLOSE(m_src_loc);
    DBMS_DATA_MINING.IMPORT_ONNX_MODEL('ALL_MINILM_L12_V2', m_blob, JSON('{"function":"embedding", "embeddingOutput":"embedding", "input":{"input": ["DATA"]}}'));
    DBMS_LOB.FREETEMPORARY(m_blob);
END;
/
```

# 🗃️ Creating and Populating the Content Article Table

## 9️⃣ Load CSV Content into SQLcl

```sql
LOAD CONTENT_ARTICLE '/Users/mvrajakishore/Downloads/data.csv' NEW;
```

## 🔟 Add Vector Column to the Table

```sql
ALTER TABLE CONTENT_ARTICLE ADD (
    CONTENT_ARTICLE_VECTOR VECTOR
);
```

## 1️⃣1️⃣ Populate the Vector Column

```sql
UPDATE CONTENT_ARTICLE
SET CONTENT_ARTICLE_VECTOR = VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING ABSTRACT AS DATA);
COMMIT;
```

# 🔍 Performing Vector Search

## 1️⃣2️⃣ Using PL/SQL for Vector Search

```sql
DECLARE search_text VARCHAR2(100) := 'Deep Learning';
BEGIN
    FOR rec IN (
        SELECT title, abstract
        FROM CONTENT_ARTICLE
        ORDER BY VECTOR_DISTANCE(
            content_article_vector,  
            VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING search_text AS DATA)
        )
        FETCH FIRST 5 ROWS ONLY
    ) LOOP
        DBMS_OUTPUT.PUT_LINE(rec.title);
    END LOOP;
END;
/
```

## 1️⃣3️⃣ Using SQL Query for Vector Search

```sql
VARIABLE search_text VARCHAR2(100);
EXEC :search_text := 'Deep Learning';
SELECT VECTOR_DISTANCE(content_article_vector, VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING :search_text AS DATA)) AS distance,
       title,
       abstract
FROM CONTENT_ARTICLE
ORDER BY 1
FETCH APPROXIMATE FIRST 5 ROWS ONLY;
```

# 🤖 Implementing RAG (Retrieval-Augmented Generation)

## 🔹 RAG = Retrieval + Generation

### **1️⃣ Retrieval**

- Store text as **vector embeddings** in a **vector database** (e.g., Oracle, Pinecone, FAISS).
- Perform a **vector similarity search** to fetch the most relevant data.

### **2️⃣ Augmentation**

- **Enhance LLM responses** by **adding retrieved data** as context.
- Example:

    ```
    "Using this context: <retrieved articles>, answer: <user question>"
    ```

### **3️⃣ Generation**

- The **LLM generates responses** based on both **its knowledge** and the **retrieved data**.

### 🔥 Key Insight

RAG enhances **LLM capabilities** by **combining retrieval + generation dynamically**.

# 🚀 RAG Implementation in Python

## 1️⃣ Install Required Libraries

```bash
pip install python-dotenv cohere oracledb
```

## 2️⃣ Create a `.env` File to Store API Keys and Database Credentials

## 3️⃣ Connect to Oracle 23AI

```python
import getpass
import os
from dotenv import load_dotenv
import oracledb
import cohere

load_dotenv()

cohere_api = os.environ.get("COHERE_API_KEY")
pw = os.environ.get("PYTHON_PASSWORD")
dbuser = os.environ.get("DB_USER")
conn_str = os.environ.get("CONNECT_STRING")

connection = oracledb.connect(
    user=dbuser,
    password=pw,
    dsn=conn_str
)
print("Successfully connected to Oracle Database")
cursor = connection.cursor()
```

## 4️⃣ Retrieve Records Using Vector Embeddings

```python
user_question = "Can you explain Thermodynamics expansion?"
query = """
SELECT title, abstract
FROM CONTENT_ARTICLE
ORDER BY VECTOR_DISTANCE(
    content_article_vector,
    VECTOR_EMBEDDING(ALL_MINILM_L12_V2 USING :query_text AS DATA)
)
FETCH FIRST 3 ROWS ONLY
"""
cursor.execute(query, {'query_text': user_question})
relevant_articles = cursor.fetchall()

retrieved_context = "\n\n".join([f"{title}: {abstract}" for title, abstract in relevant_articles])
co = cohere.Client(cohere_api)

if retrieved_context.strip():
    response = co.chat(
        model="command-r-plus",
        message=f"Using this context:\n{retrieved_context}\n\nAnswer the question: {user_question}"
    )
    print(response.text)
else:
    print("No relevant articles found.")
```

## 5️⃣ Close the Database Connection

```python
cursor.close()
connection.close()
```
