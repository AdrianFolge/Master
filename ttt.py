from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
import os
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from datasets import load_dataset
import pandas as pd

loader = DirectoryLoader('Master\data', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

os.environ["OPENAI_API_KEY"] = "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"
paraphrase = SentenceTransformerEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2') 
e5 = SentenceTransformerEmbeddings(model_name='intfloat/multilingual-e5-large')
OpenAIEmbed = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = [paraphrase, e5, OpenAIEmbed]
db1 = Chroma.from_documents(docs, paraphrase)
db2 = Chroma.from_documents(docs, e5)
db3 = Chroma.from_documents(docs, OpenAIEmbed)