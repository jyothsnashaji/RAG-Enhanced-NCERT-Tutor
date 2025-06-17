from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from typing import List

from fastapi.responses import JSONResponse
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables import RunnableSequence 
from prompts import explain_prompt, general_prompt
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
import pdb
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
load_dotenv()


explain_chains = {}
session_histories = {}
translation_chains = {}
exercise_chains = {}
general_chains = {}

def get_session_history(session_id):
    if session_id not in session_histories:
        session_histories[session_id] = InMemoryChatMessageHistory()
    return session_histories[session_id]

class ExplainDestinationChain():
    def __init__(self, session_id):
        self.session_id = session_id
        self.llm =  ChatOpenAI(model="gpt-4o-mini")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vectorstore = FAISS.load_local("/Users/jshaji/Library/CloudStorage/OneDrive-Cisco/IISc/Deep Learning/RAG-Enhanced-NCERT-Tutor/ncert_tutor/vector_store", self.embeddings, allow_dangerous_deserialization=True)
        self.retriever = self.vectorstore.as_retriever()
        self._chain = explain_prompt | self.llm
      
                                
    def get_rag_response(self, query: str, grade: str = "", subject: str = ""):
        try:
            print(f"[QUERY] Grade: {grade}, Subject: {subject}, Question: {query}")


            # Step 1: Final prompt with static variables filled in
            final_prompt = explain_prompt.partial(grade=grade, subject=subject)

            # Step 2: Combine retrieved documents into string context
            combine_docs_chain = create_stuff_documents_chain(self.llm, final_prompt)
            # Step 3: Create RAG chain
            rag_chain = create_retrieval_chain(self.retriever, combine_docs_chain)

            # Step 4: Add chat memory
            rag_chain_with_memory = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history"
            )

            # Step 5: Run it
            result = rag_chain_with_memory.invoke(
                {"input": query, "chat_history": get_session_history(self.session_id).messages},
                config={"configurable": {"session_id": self.session_id}}
            )

            print("=== RAG Chain Result ===")
            print(result)

            # Extract the final answer (always under key "answer")
            raw_answer = result.get("answer", "No answer found.")
            answer = raw_answer.get("answer") if isinstance(raw_answer, dict) else raw_answer

            # Extract sources
            simplified_sources = []
            for doc in result.get("context", []):
                metadata = doc.metadata
                page_content = doc.page_content if hasattr(doc, 'page_content') else "No content available"
                print(f"Source doc metadata: {metadata}")
                simplified_sources.append(DocumentInfo(
                    page=str(metadata.get("page", "#")),
                    file=str(metadata.get("source", "Untitled")),
                    snippet=page_content[:300]  # First 300 characters
                ))
                    

            return Response(
                answer=answer,
                retrieved_documents=simplified_sources
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[ERROR] RAG pipeline failed: {e}")
            return Response(
                answer="Sorry, an error occurred while generating your answer.",
                retrieved_documents=[]
            )

class TranslationDestinationChain():
    def __init__(self, session_id):
        self.session_id = session_id
        # Define translation chain here if needed

class ExerciseDestinationChain():
    def __init__(self, session_id):
        self.session_id = session_id
        # Define exercise chain here if needed
    
class GeneralDestinationChain():
    def __init__(self, session_id):
        self.session_id = session_id
        self.llm = ChatGroq(model="llama3-8b-8192") 
        self._chain = general_prompt | self.llm
        
        # Define general chain here if needed

class DestinationChainArgs():
    def __init__(self, destination, keyword, target_language, query):
        self.destination = destination
        self.keyword = keyword
        self.target_language = target_language
        self.query = query

class Quiz(BaseModel):
    question: str
    options: List[str]
    answer: str

class DocumentInfo(BaseModel):
    page: str
    file: str
    snippet: str

class Response(BaseModel):
    answer: str
    retrieved_documents: List[DocumentInfo]
    quiz: Quiz = None  # Optional field for quiz data



def get_explaination_chain(session_id):
    if session_id not in explain_chains:
        explain_chains[session_id] = ExplainDestinationChain(session_id)
    return explain_chains[session_id]

def get_translation_chain(session_id):
    if session_id not in translation_chains:
        translation_chains[session_id] = TranslationDestinationChain(session_id)
    return translation_chains[session_id]

def get_exercise_chain(session_id):
    if session_id not in exercise_chains:
        exercise_chains[session_id] = ExerciseDestinationChain(session_id)
    return exercise_chains[session_id]

def get_general_chain(session_id):
    if session_id not in general_chains:
        general_chains[session_id] = GeneralDestinationChain(session_id)
    return general_chains[session_id]

def run_general_chain(arguments, session_id):
    chain = get_general_chain(session_id)
    answer =  chain._chain.invoke(
        {"input": arguments.query, "chat_history": get_session_history(session_id).messages},
        config={"configurable": {"session_id": session_id}}
    )
    return Response(answer=answer.content, retrieved_documents=[]) # Assuming no sources for general queries


def run_explaination_chain(arguments: DestinationChainArgs, session_id: str):
    """
    Run the explaination chain with the provided arguments and session ID.
    Inputs: DestinationChainArgs object containing:
        - destination: The type of destination (e.g., "explain", "translate", etc.)
        - keyword: The keyword to search for in the context
        - target_language: The language to translate to (if applicable)
        - question: The question or query to be answered
    Outputs: Response object containing:
        - answer: The generated answer from the explaination chain
        - retrieved_documents: List of DocumentInfo objects containing source information
    """
    chain = get_explaination_chain(session_id)
    return chain.get_rag_response(arguments.keyword)

def run_translation_chain(arguments, session_id):
    """Run the translation chain with the provided arguments and session ID.
    Inputs: DestinationChainArgs object containing:
        - destination: The type of destination (e.g., "explain", "translate", etc.)
        - keyword: The keyword to search for in the context
        - target_language: The language to translate to (if applicable)
        - question: The question or query to be answered
    Outputs: Response object containing:
        - answer: The generated answer from the translation chain
        - retrieved_documents: List of DocumentInfo objects containing source information
    """
    return Response(
        answer="Translation functionality is not yet implemented.",
        retrieved_documents=[]
    )  # Placeholder response until translation chain is implemented

def run_exercise_chain(arguments, session_id):
    """Run the exercise chain with the provided arguments and session ID.
    Inputs: DestinationChainArgs object containing:
        - destination: The type of destination (e.g., "explain", "translate", etc.)
        - keyword: The keyword to search for in the context
        - target_language: The language to translate to (if applicable)
        - question: The question or query to be answered
    Outputs: Response object containing:
        - answer: The generated answer from the exercise chain
        - retrieved_documents: List of DocumentInfo objects containing source information
        - quiz: Quiz object containing generated questions and answers
        
    """
    return Response(
        answer="Exercise functionality is not yet implemented.",
        retrieved_documents=[],
        quiz=Quiz(
            question="This is a placeholder question",
            options=["Option 1", "Option 2", "Option 3", "Option 4"],
            answer="Option 1"
        )
    )  # Placeholder response until exercise chain is implemented