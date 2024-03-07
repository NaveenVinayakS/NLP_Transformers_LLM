from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from config import *

class EduBotCreator:

    def __init__(self):
        self.prompt_temp = PROMPT_TEMPLATE
        self.input_variables = INP_VARS
        self.chain_type = CHAIN_TYPE
        self.search_kwargs = SEARCH_KWARGS
        self.embedder = EMBEDDER
        self.vector_db_path = VECTOR_DB_PATH
        self.model_ckpt = MODEL_CKPT
        self.model_type = MODEL_TYPE
        self.max_new_tokens = MAX_NEW_TOKENS
        self.temperature = TEMPERATURE

    def create_custom_prompt(self):
        custom_prompt_temp = PromptTemplate(template=self.prompt_temp,
                            input_variables=self.input_variables)
        return custom_prompt_temp
    
    def load_llm(self):
        llm = CTransformers(
                model = self.model_ckpt,
                model_type=self.model_type,
                max_new_tokens = self.max_new_tokens,
                temperature = self.temperature
            )
        return llm
    
    def load_vectordb(self):
        hfembeddings = HuggingFaceEmbeddings(
                            model_name=self.embedder, 
                            model_kwargs={'device': 'cpu'}
                        )

        vector_db = FAISS.load_local(self.vector_db_path, hfembeddings)
        return vector_db

    def create_bot(self, custom_prompt, vectordb, llm):
        retrieval_qa_chain = RetrievalQA.from_chain_type(
                                llm=llm,
                                chain_type=self.chain_type,
                                retriever=vectordb.as_retriever(search_kwargs=self.search_kwargs),
                                return_source_documents=True,
                                chain_type_kwargs={"prompt": custom_prompt}
                            )
        return retrieval_qa_chain
    
    def create_edubot(self):
        self.custom_prompt = self.create_custom_prompt()
        self.vector_db = self.load_vectordb()
        self.llm = self.load_llm()
        self.bot = self.create_bot(self.custom_prompt, self.vector_db, self.llm)
        return self.bot
    

'''
retriever=vectordb.as_retriever(search_kwargs=self.search_kwargs),

search_kwargs={'k': 2}: This is an argument passed to the as_retriever() method. It specifies search parameters for the retriever. 
In this case, k is set to 2, which likely indicates that the retriever should retrieve the top 2 most relevant documents or passages.'''

'''
The process typically involves using similarity search techniques. Here's how it generally works:

Embedding the Documents: Each document or passage in the vector database is first converted into a fixed-length vector representation using an embedding model. This embedding represents the semantic content of the document in a high-dimensional vector space.

Query Embedding: When a query is received, it is also converted into a fixed-length vector representation using the same embedding model. This vector represents the semantic content of the query.

Similarity Calculation: The cosine similarity (or another similarity metric) is often used to measure the similarity between the query vector and each document vector in the database. Cosine similarity measures the cosine of the angle between two vectors and indicates how similar they are in direction.

Ranking: The documents are then ranked based on their similarity to the query. The documents with the highest cosine similarity scores are considered the most relevant to the query.

Top-k Retrieval: Finally, the top-k documents (where k is typically specified, as in search_kwargs={'k': 2}) with the highest similarity scores are selected and returned as the most relevant chunks or passages.

'''