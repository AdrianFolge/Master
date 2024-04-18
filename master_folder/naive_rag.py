import re
from deep_translator import GoogleTranslator
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from datasets import load_dataset

def naive_rag_translated(instances, file_path):
    loader = DirectoryLoader('../data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    databases = {}
    for doc in documents:
        source = doc.metadata['source']
        match = re.search(r'\/([A-Za-z_]+)\.pdf', source)
        if match:
            municipality_name = match.group(1)
        docs = text_splitter.split_documents([doc])
        for document in docs:
            page_content = document.page_content
            translated_content = GoogleTranslator(source='no', target='en').translate(text=page_content)
            document.page_content = translated_content
        for index, doc in enumerate(docs):
            if isinstance(doc.page_content, type(None)):
                docs[index].page_content = ""
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        db = FAISS.from_documents(docs, embeddings)
        databases[municipality_name] = db
    list_of_answers = []

    for num in range(instances):
        query = references["spørsmål"][num]
        translated_query = GoogleTranslator(source='no', target='en').translate(text=query)
        kommunenavn = references["kommunenavn"][num]
        db = databases[kommunenavn]
        found_docs = db.similarity_search(translated_query)
        context = found_docs[0].page_content
        translated_answer = GoogleTranslator(source='en', target='no').translate(text=context)
        list_of_answers.append(translated_answer)
    return list_of_answers

def naive_rag(instances, file_path):
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]") 
    loader = DirectoryLoader('../data/', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    databases = {}
    for doc in documents:
        source = doc.metadata['source']
        match = re.search(r'\/([A-Za-z_]+)\.pdf', source)
        if match:
            municipality_name = match.group(1)
        docs = text_splitter.split_documents([doc])
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        db = FAISS.from_documents(docs, embeddings)
        databases[municipality_name] = db
        list_of_answers = []

    for num in range(instances):
        query = references["spørsmål"][num]
        kommunenavn = references["kommunenavn"][num]
        db = databases[kommunenavn]
        found_docs = db.similarity_search(query)
        context = found_docs[0].page_content
        list_of_answers.append(context)
    return list_of_answers