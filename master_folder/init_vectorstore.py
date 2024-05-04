import re
from deep_translator import GoogleTranslator
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

def init_vectorstore(embeddings, text_splitter, translate=False):
    loader = DirectoryLoader('../data', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    databases = {}
    for doc in documents:
        source = doc.metadata['source']
        match = re.search(r'\/([A-Za-z_]+)\.pdf', source)
        if match:
            municipality_name = match.group(1)
        docs = text_splitter.split_documents([doc])
    
        for document in docs:
            if translate:
                page_content = document.page_content
                translated_content = GoogleTranslator(source='no', target='en').translate(text=page_content)
                document.page_content = translated_content
            else:
                pass  # If no translation is needed, leave the content as is
    
        # Handling None type content
        for index, doc in enumerate(docs):
            if isinstance(doc.page_content, type(None)):
                docs[index].page_content = ""

        db = FAISS.from_documents(docs, embeddings)
        databases[municipality_name] = db
    return databases

def init_semantic_vectorstore(translate=False):
    text_splitter = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
    )
    loader = DirectoryLoader('../data', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    databases = {}
    for doc in documents:
        source = doc.metadata['source']
        match = re.search(r'\/([A-Za-z_]+)\.pdf', source)
        if match:
            municipality_name = match.group(1)
        docs = text_splitter.split_documents([doc])
        for document in docs:
            if translate:
                page_content = document.page_content
                translated_content = GoogleTranslator(source='no', target='en').translate(text=page_content)
                document.page_content = translated_content
            else:
                pass  # If no translation is needed, leave the content as is
        # Handling None type content
        for index, doc in enumerate(docs):
            if isinstance(doc.page_content, type(None)):
                docs[index].page_content = ""

        db = FAISS.from_documents(docs, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
        databases[municipality_name] = db
    return databases


