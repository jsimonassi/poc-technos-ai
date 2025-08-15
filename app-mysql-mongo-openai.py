import os
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import mysql.connector
from openai import OpenAI
from pymongo import MongoClient
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ==========================================
# Configuração inicial da API
# ==========================================
app = FastAPI()

# Inicializa cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicializa embeddings
embeddings_model = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")

# Conexão Mongo
mongo_client = MongoClient(f"mongodb://{os.getenv('MONGO_HOST')}:{os.getenv('MONGO_PORT', 27017)}")
mongo_db = mongo_client[os.getenv('MONGO_DB', 'mormaiismartwatches')]
mongo_collection = mongo_db["faq_ai_embeddings"]

# ==========================================
# Função para carregar FAQs do MySQL
# ==========================================
def load_faq_mysql():
    db_config = {
        "host": os.getenv("MYSQL_HOST", "smartbackend-mysql-1"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "mormaiismartwatches"),
    }

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT topic, content FROM faqs")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    documents = []
    for row in rows:
        pergunta_norm = row['topic'].strip().lower()
        content = f"Pergunta: {row['topic']}\nResposta: {row['content']}"
        documents.append(Document(
            page_content=content,
            metadata={"pergunta": pergunta_norm}
        ))
    return documents

# ==========================================
# Endpoint para atualizar embeddings
# ==========================================
@app.post("/update_embeddings")
async def update_embeddings():
    documents = load_faq_mysql()
    if not documents:
        return JSONResponse({"error": "Não há FAQs no MySQL para gerar embeddings."}, status_code=404)

    for doc in documents:
        vector = embeddings_model.embed_query(doc.page_content)  # lista de floats
        mongo_collection.update_one(
            {"pergunta": doc.metadata["pergunta"]},
            {
                "$set": {
                    "content": doc.page_content,
                    "embedding": vector  # salva como lista de floats
                }
            },
            upsert=True
        )

    return {"message": f"Embeddings atualizados para {len(documents)} FAQs."}

# ==========================================
# Carrega embeddings do MongoDB
# ==========================================
def load_embeddings_from_mongo():
    documents = []
    embeddings_list = []

    cursor = mongo_collection.find()
    for doc in cursor:
        documents.append(Document(page_content=doc["content"], metadata={"pergunta": doc["pergunta"]}))
        embeddings_list.append(doc["embedding"])  # já é lista de floats

    if not documents:
        return None, None

    # Monta lista de tuplas (texto, vetor)
    text_embeddings = [(doc.page_content, emb) for doc, emb in zip(documents, embeddings_list)]

    # Cria FAISS com metadados
    vectorstore = FAISS.from_embeddings(
        text_embeddings,
        embedding=embeddings_model,
        metadatas=[doc.metadata for doc in documents]
    )

    return documents, vectorstore

# Inicializa variáveis globais
documents, vectorstore = load_embeddings_from_mongo()

# ==========================================
# Endpoint principal de perguntas
# ==========================================
@app.post("/question")
async def responder_pergunta(request: Request):
    global documents, vectorstore

    if not documents or not vectorstore:
        return JSONResponse({"error": "Embeddings não encontrados. Execute /update_embeddings primeiro."}, status_code=404)

    start_time = time.perf_counter()
    data = await request.json()
    pergunta = data.get("pergunta", "").strip()

    if not pergunta:
        return JSONResponse({"error": "Envie a pergunta no formato JSON com chave 'pergunta'"}, status_code=400)

    pergunta_norm = pergunta.lower()
    contexto = None
    score_max = 1.0

    # Match exato
    for doc in documents:
        if pergunta_norm == doc.metadata["pergunta"]:
            contexto = doc.page_content
            score_max = 0.0
            break

    # Busca vetorial
    if contexto is None:
        docs_scores = vectorstore.similarity_search_with_score(pergunta, k=10)
        if not docs_scores:
            return {"resposta": "Não tenho informações suficientes no momento para responder a essa pergunta."}

        score_max = docs_scores[0][1] / 100
        if score_max > 0.3:
            return {"resposta": "Não tenho informações suficientes no momento para responder a essa pergunta."}

        docs_ordenados = sorted(docs_scores, key=lambda x: x[1])
        docs_filtrados = [doc for doc, _ in docs_ordenados[:3]]
        contexto = "\n\n".join(doc.page_content for doc in docs_filtrados)

    prompt = f"""
Você é um assistente especialista em smartwatches da marca Mormaii Smartwatches.
Responda usando **apenas** as informações presentes entre as tags <contexto> e </contexto>.
Se não encontrar a resposta no contexto, responda exatamente:
"Não tenho informações suficientes no momento para responder a essa pergunta."

<contexto>
{contexto}
</contexto>

Pergunta: {pergunta}

Resposta:
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    resposta = completion.choices[0].message.content.strip()
    elapsed_time = time.perf_counter() - start_time
    return {
        "resposta": resposta,
        "tempo_execucao_segundos": round(elapsed_time, 3)
    }
