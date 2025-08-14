import os
import csv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import httpx

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# ==========================================
# Configuração inicial da API
# ==========================================
app = FastAPI()

# URL do servidor Ollama (LLM local ou remoto)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

# Caminho do CSV contendo FAQ
CSV_PATH = os.getenv("CSV_PATH", "faq_smartwatch.csv")

# Quantidade inicial de documentos buscados
K = 10  

# Limiar de confiança (quanto menor o score, mais parecido)
SCORE_THRESHOLD = 0.30  

# Quantos documentos enviar no contexto final
TOP_N = 3

# ==========================================
# Função para carregar o CSV de FAQ
# ==========================================
def load_faq_csv(path: str):
    documents = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            pergunta_norm = row['topic'].strip().lower()
            content = f"Pergunta: {row['topic']}\nResposta: {row['content']}"
            documents.append(Document(
                page_content=content,
                metadata={"pergunta": pergunta_norm}
            ))
    return documents

# ==========================================
# Carrega documentos do CSV
# ==========================================
documents = load_faq_csv(CSV_PATH)

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
    data = await request.json()
    pergunta = data.get("pergunta", "").strip()

    if not pergunta:
        return {"error": "Envie a pergunta no formato JSON com chave 'pergunta'"}

    pergunta_norm = pergunta.lower()

    contexto = None
    score_max = 1.0  # Score alto = menos relevante

    for doc in documents:
        if pergunta_norm == doc.metadata["pergunta"]:
            contexto = doc.page_content
            score_max = 0.0  # Score perfeito para match exato
            print("🔍 Contexto obtido por match exato.")
            break

    # 2. Busca vetorial (somente se não encontrou match exato)
    if contexto is None:
        docs_scores = vectorstore.similarity_search_with_score(pergunta, k=K)
        print(f"🔍 Encontrados {len(docs_scores)} documentos relevantes na busca vetorial.")
        if not docs_scores:
            return {"resposta": "Não tenho informações suficientes no momento para responder a essa pergunta."}
        print(f"🔍 Pontuações dos documentos: {[score for _, score in docs_scores]}")
        score_max = docs_scores[0][1]/100  # Normaliza o score para o intervalo [0, 1]
        print(f"🔍 Score máximo encontrado: {score_max}. SCORE_THRESHOLD: {SCORE_THRESHOLD}") 
        # Filtro por limiar de confiança
        if score_max > SCORE_THRESHOLD:
            return {"resposta": "Não tenho informações suficientes no momento para responder a essa pergunta."}

        # Seleciona os TOP_N mais relevantes
        docs_ordenados = sorted(docs_scores, key=lambda x: x[1])
        docs_filtrados = [doc for doc, _ in docs_ordenados[:TOP_N]]
        contexto = "\n\n".join(doc.page_content for doc in docs_filtrados)
        print("🔍 Contexto obtido por busca vetorial.")

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
    
    async def event_stream():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", OLLAMA_URL, json={
                "model": "deepseek-r1:1.5b",
                "prompt": prompt,
                "stream": False
            }) as resp:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield line + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")
