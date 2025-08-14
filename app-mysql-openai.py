import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time
from openai import OpenAI
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import mysql.connector

# ==========================================
# Configuração inicial da API
# ==========================================
app = FastAPI()

# Inicializa cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def load_faq_mysql():
    print("ENVS: ", os.getenv("MYSQL_HOST"), os.getenv("MYSQL_USER"), os.getenv("MYSQL_PASSWORD"), os.getenv("MYSQL_DATABASE"))
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

    documents = []
    for row in rows:
        pergunta_norm = row['topic'].strip().lower()
        content = f"Pergunta: {row['topic']}\nResposta: {row['content']}"
        documents.append(Document(
            page_content=content,
            metadata={"pergunta": pergunta_norm}
        ))

    cursor.close()
    conn.close()
    return documents


# Quantidade inicial de documentos buscados
K = 10  

# Limiar de confiança (quanto menor o score, mais parecido)
SCORE_THRESHOLD = 0.30  

# Quantos documentos enviar no contexto final
TOP_N = 3

documents = load_faq_mysql()

# ==========================================
# Inicializa embeddings
# ==========================================
embeddings = HuggingFaceEmbeddings(model_name="multi-qa-mpnet-base-dot-v1")
vectorstore = FAISS.from_documents(documents, embeddings)

# ==========================================
# Endpoint principal de perguntas
# ==========================================
@app.post("/question")
async def responder_pergunta(request: Request):
    start_time = time.perf_counter()
    data = await request.json()
    pergunta = data.get("pergunta", "").strip()

    if not pergunta:
        return JSONResponse({"error": "Envie a pergunta no formato JSON com chave 'pergunta'"}, status_code=400)

    pergunta_norm = pergunta.lower()
    contexto = None
    score_max = 1.0

    # 1. Match exato
    for doc in documents:
        if pergunta_norm == doc.metadata["pergunta"]:
            contexto = doc.page_content
            score_max = 0.0
            print(" Contexto obtido por match exato.")
            break

    # 2. Busca vetorial
    if contexto is None:
        docs_scores = vectorstore.similarity_search_with_score(pergunta, k=K)
        print(f" Encontrados {len(docs_scores)} documentos relevantes na busca vetorial.")
        if not docs_scores:
            return {"resposta": "Não tenho informações suficientes no momento para responder a essa pergunta."}

        print(f" Pontuações dos documentos: {[score for _, score in docs_scores]}")
        score_max = docs_scores[0][1] / 100
        print(f" Score máximo encontrado: {score_max}. SCORE_THRESHOLD: {SCORE_THRESHOLD}")

        if score_max > SCORE_THRESHOLD:
            return {"resposta": "Não tenho informações suficientes no momento para responder a essa pergunta."}

        docs_ordenados = sorted(docs_scores, key=lambda x: x[1])
        docs_filtrados = [doc for doc, _ in docs_ordenados[:TOP_N]]
        contexto = "\n\n".join(doc.page_content for doc in docs_filtrados)
        print(" Contexto obtido por busca vetorial.")

    # Prompt com contexto delimitado
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

    print(f"Prompt enviado ao LLM:\n{prompt}")

    # Chamada para a API da OpenAI
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
