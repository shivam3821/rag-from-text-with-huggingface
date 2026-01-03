from langchain_chroma import Chroma
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

persistent_directory = "db/chroma_db"

# Load Embeddings and vector store
embedding_model = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

# Query
# query = "What was NVIDIA's first graphic accelerator called?"
# query = "Which company did NVIDIA acquire to enter the mobile processor market?"
# query = "What was Microsoft's first hardware product release?"
# query = "How much did Microsoft pay to acquire GitHub?"
# query = "In what year did Tesla begin production of the Roadster?"
# query = "Who succeeded Ze'ev Drori as CEO in October 2008?"
query = "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# query = "Which island does SpaceX lease for its launches in the Pacific?"
# query = "What was the original name of Microsoft before it became Microsoft?" #hallucinating

retriever = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs={
        "k":3,                  # top 3 chunks
        "score_threshold": 0.3  # only retrieve chunk with cosine similarity >= 0.3
        }
)


relevant_docs = retriever.invoke(query)

# Display Results
# print("---Context---")
# for i, doc in enumerate(relevant_docs, 1):
#     print("Document {}, \n{}\n".format(i,doc.page_content.replace("\n", " ")))  #our txt file has line break: we are replacing the linebreak with spaces.

# Combine the query and the relevant document contents
prompt = f"""You are a helpful assistant. Based on the following documents, answer the questio:

Question: {query}

Documents: 
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

"""

# Generation Model
llm = HuggingFaceEndpoint(
    model="meta-llama/Llama-3.2-3B-Instruct",
    task="text-generation",
    max_new_tokens=50,
    temperature=0.7,
)

llm = ChatHuggingFace(llm=llm)

# Run Model
result = llm.invoke([
    HumanMessage(content=prompt)
])


# Display the full result and content only
print("\n---Generated Response---")
print(result.content)


# Synthetic Questions:

# 1. "What was NVIDIA's first graphic accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "Which island does SpaceX lease for its launches in the Pacific?"
# 9. "What was the original name of Microsoft before it became Microsoft?"