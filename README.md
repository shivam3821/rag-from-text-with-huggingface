# rag-from-text-with-huggingface
A Retrieval-Augmented Generation system built to answer queries from text documents. It leverages Hugging Face open-source embedding models and language models, integrates ChromaDB as the vector database, and uses LangChain for building the retrieval workflow.


# Step 1: Create a Virtual Environment

Our virtual environment name is RAG-env

```sh
python -m venv RAG-ENV
RAG-ENV\Scripts\activate
```



## Step 2: Dependencies

```sh

pip install langchain
pip install langchain-community
pip install langchain-text-splitters
pip install langchain-chroma
pip install langchain-huggingface
pip install python-dotenv
pip install sentence-transformers
pip install huggingface-hub
pip install ipykernel


```

$$ OR $$

Create a `requirement.txt` file with all the dependencies mention and install all at once..

```sh
pip install -r requirements.txt
```

- `langchain` : core framework 
- `langchain-community` : forintegrations for community models, loaders, and vector stores.
- `langchain-text-splitters` : for chinking
- `langchain-chroma` : vector database
- `python-dotenv` : for securly saving api keys
- `sentense-transformers` : embedding model
- `langchain-huggingface` : LangChain integration for Hugging Face models
- `huggingface-hub` : library for accessing and managing Hugging Face models and datasets
- `ipkernel` : to run jupyter notebook



## Step 3: Import all Libraries

```py
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()
```


## Step 4: Load Document

The list of LangChain document will look like this

```txt
documents = [
    Google{
        page_content = "All the document content
        metadata={'source':'docs/google.txt}
    },
    microsoft{
        page_content = "All the document content
        metadata={'source':'docs/microsoft.txt}
    },
    etc...
]
```


### Embedding Model

```py
embedding_model = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
    )

```

### Persistent Directory

> **Note:** After creating the Embedding -> Vector Database is created and a new folder will be created where the data will be stored ``RAG_Project/db/chroma_db``

### Generation Model

```py
HuggingFaceEndpointEmbeddings(
    model="gpt2",
    max_new_tokens=100,
    temperature=0.7,
)
```

1. `max_new_tokens`
- No. of tokens needed as output
- 1 token = 3-4 characters
- 100 tokens = 70-80 words

2. `temperature`
- This controls creativity vs. determinism.
- Higher temperature → more random / creative
- Lower temperature → more focused / predictable

| Temperature | Output style                          |
| ----------- | ------------------------------------- |
| 0.0         | deterministic, same answer every time |
| 0.3         | factual, focused                      |
| 0.7         | balanced (default)                    |
| 1.0         | creative, varied                      |
| >1.2        | very random, sometimes nonsense       |





