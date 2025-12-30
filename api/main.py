import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# Charger les variables d’environnement depuis .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Connexion MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot_db"]
history_collection = db["conversations"]

# Embeddings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

# Chargement vectorstore FAISS
vectorstore = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)

# LLM OpenRouter
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",
    temperature=0,
    openai_api_base=os.environ["OPENAI_API_BASE"]
)

# Mémoire conversationnelle
memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

# Chaîne de QA avec mémoire
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
)

# FastAPI app
app = FastAPI()

class Query(BaseModel):
    user_id: str
    question: str

def save_conversation(user_id, question, answer, source="chat"):
    answer_text = answer.content if isinstance(answer, AIMessage) else str(answer)
    history_collection.insert_one({
        "user_id": user_id,
        "source": source,
        "question": question,
        "answer": answer_text.strip(),
        "timestamp": datetime.now(timezone.utc)
    })

@app.post("/chat")
def chat_normal(query: Query):
    try:
        human_msg = HumanMessage(content=query.question)
        response = llm.invoke([human_msg])
        save_conversation(query.user_id, query.question, response, source="chat")
        return {"response": response.content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/chat_with_knowledge")
def chat_knowledge(query: Query):
    try:
        chat_history = memory.load_memory_variables({}).get("chat_history", [])
        response = qa_chain.invoke({
            "question": query.question,
            "chat_history": chat_history
        })
        
        # Sauvegarde de la conversation
        save_conversation(query.user_id, query.question, response["answer"], source="chat_with_knowledge")
        
        # Auto-résumé de la réponse
        original_text = response["answer"].strip()
        prompt = (
            "Fais un résumé concis, simple et clair de cette réponse en 2 à 3 phrases max, "
            "sans répéter les mêmes formulations :\n\n"
            f"{original_text}"
        )
        summary_response = llm.invoke([HumanMessage(content=prompt)])
        summary_text = summary_response.content

        # Sauvegarde du résumé dans MongoDB
        save_summary(query.user_id, original_text, summary_text)

        return {
            "response": original_text,
            "summary": summary_text
        }

    except Exception as e:
        return {"error": str(e)}

class SummaryRequest(BaseModel):
    user_id: str
    text: str = None

def save_summary(user_id, original_text, summary_text):
    history_collection.insert_one({
        "user_id": user_id,
        "source": "summary",
        "original_text": original_text.strip(),
        "summary": summary_text.strip(),
        "timestamp": datetime.now(timezone.utc)
    })

@app.post("/summarize")
def summarize(summary_req: SummaryRequest):
    try:
        if summary_req.text:
            original_text = summary_req.text.strip()
            prompt = (
                "Fais un résumé concis, simple et clair de ce texte en 2 à 3 phrases max, "
                "en ne répétant pas les mêmes formulations :\n\n"
                f"{original_text}"
            )
        else:
            last_chat = history_collection.find_one(
                {"user_id": summary_req.user_id},
                sort=[("timestamp", -1)]
            )
            if not last_chat or "answer" not in last_chat:
                return {"error": "No previous conversation found for this user."}
            original_text = last_chat["answer"].strip()
            prompt = (
                "Fais un résumé concis, simple et clair de cette réponse précédente en 2 à 3 phrases max, "
                "sans reprendre les phrases mot à mot :\n\n"
                f"{original_text}"
            )

        response = llm.invoke([HumanMessage(content=prompt)])
        summary_text = response.content

       
        save_summary(summary_req.user_id, original_text, summary_text)

        return {"summary": summary_text}
    except Exception as e:
        return {"error": str(e)}