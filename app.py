from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from collections import defaultdict

from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain.tools import Tool
from langchain.schema import Document
import pdb 

# -------------------- FastAPI Setup --------------------
app = FastAPI()

# -------------------- Pydantic Models --------------------
class Query(BaseModel):
    session_id: str
    question: str

class DocumentInfo(BaseModel):
    page: str = ""
    link: str = ""
    snippet: str

class Response(BaseModel):
    answer: str
    retrieved_documents: List[DocumentInfo]

# -------------------- Vector DB --------------------
def load_vector_db(vector_db_path: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=vector_db_path, embedding_function=embeddings)

vector_db_path = "vector_db4"
vector_db = load_vector_db(vector_db_path)
print("✅ Vector DB loaded")

# -------------------- LLM & Retriever --------------------
groq_api_key = "API_KEY"
llm = ChatGroq(groq_api_key=groq_api_key, model="llama3-8b-8192")

# Define system prompt
system_prompt = SystemMessage(content="""
You are a friendly teacher.
- If any query related to physics is asked or a follow up to previous query. Load context from tools using retrieve_physics_docs_with_tracking
- Use the loaded context give explainations and clarify doubts.
- Mention chapter name but not figure names.
- If you do not find relevant context, politely say it’s not in the syllabus.
- For non-physics or casual questions, you may respond freely.
""")

# Setup memory and inject system prompt
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


session_memories = {}

def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_memories:
        # New session, create fresh memory
        session_memories[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        # Optionally add your system prompt here for every new memory
        session_memories[session_id].chat_memory.add_message(system_prompt)
    return session_memories[session_id]

# Create contextual retriever with compression
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_db.as_retriever(search_kwargs={"k": 5})
)

# -------------------- Custom Tool With Tracking --------------------
last_retrieved_docs = []

def retrieve_physics_docs_with_tracking(query: str) -> str:
    global last_retrieved_docs
    docs = compression_retriever.get_relevant_documents(query)
    last_retrieved_docs = docs
    return "\n\n".join([doc.page_content for doc in docs])

physics_tool = Tool(
    name="retrieve_physics_docs",
    func=retrieve_physics_docs_with_tracking,
    description="ALWAYS load documents and answer from them if physics related query",
)

# -------------------- Agent Setup --------------------


# -------------------- API Endpoint --------------------
@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        agent = initialize_agent(
            tools=[physics_tool],
            llm=llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=get_memory(query.session_id),
            verbose=True,
            system_prompt = system_prompt
        )

        # Run the agent to get the response
        result = agent.run(query.question)

        # Build the document list from last retrieval
        retrieved_docs_info = []
        for doc in last_retrieved_docs:
            metadata = doc.metadata or {}
            retrieved_docs_info.append(DocumentInfo(
                page=str(metadata.get("page", "")),
                link=str(metadata.get("source", "")),
                snippet=doc.page_content[:300]  # First 300 characters
            ))

        return Response(answer=result, retrieved_documents=retrieved_docs_info)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------- Local Run --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
