from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from datasets import Dataset
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

def ragas(): 
    synthetic_data_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False
    )
    loader = DirectoryLoader('../data', glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    final_dataset = []
    for doc in documents:
        docs = synthetic_data_splitter.split_documents([doc])
        db = FAISS.from_documents(docs, embedding=OpenAIEmbeddings(model="text-embedding-3-large"))
        semantic_chunk_retriever = db.as_retriever(search_kwargs={"k" : 1})
        rag_template = """\
        Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'. Give the answer in Norwegian.

        User's Query:
        {question}

        Context:
        {context}
        """

        rag_prompt = ChatPromptTemplate.from_template(rag_template)
        base_model = ChatOpenAI()
        semantic_rag_chain = (
            {"context" : semantic_chunk_retriever, "question" : RunnablePassthrough()}
            | rag_prompt
            | base_model
            | StrOutputParser()
        )
        questions = []
        ground_truths_semantic = []
        contexts = []
        answers = []

        question_prompt = """\
        You are a teacher preparing a test. Please create a question that can be answered by referencing the following context.

        Context:
        {context}
        """

        question_prompt = ChatPromptTemplate.from_template(question_prompt)

        ground_truth_prompt = """\
        Use the following context and question to answer this question using *only* the provided context.

        Question:
        {question}

        Context:
        {context}
        """

        ground_truth_prompt = ChatPromptTemplate.from_template(ground_truth_prompt)

        question_chain = question_prompt | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()
        ground_truth_chain = ground_truth_prompt | ChatOpenAI(model="gpt-3.5-turbo") | StrOutputParser()

        for chunk in docs[0:3]:
            questions.append(question_chain.invoke({"context" : chunk.page_content}))
            contexts.append([chunk.page_content])
            ground_truths_semantic.append(ground_truth_chain.invoke({"question" : questions[-1], "context" : contexts[-1]}))
            answers.append(semantic_rag_chain.invoke(questions[-1]))
        qagc_list = []
        for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths_semantic):
            qagc_list.append({
                "question" : question,
                "answer" : answer,
                "contexts" : context,
                "ground_truth" : ground_truth
            })
        eval_dataset = Dataset.from_list(qagc_list)
        result = evaluate(
            eval_dataset,
            metrics=[
                context_precision,
                faithfulness,
                answer_relevancy,
                context_recall,
            ],
        )
        final_dataset.append(result)
    return final_dataset

def ragas_with_params(answers, questions, contexts, ground_truths):
    qagc_list = []
    for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths):
        qagc_list.append({
            "question" : question,
            "answer" : answer,
            "contexts" : [context],
            "ground_truth" : ground_truth
        })
    eval_dataset = Dataset.from_list(qagc_list)
    result = evaluate(
        eval_dataset,
        metrics=[
            context_precision,
            faithfulness,
            answer_relevancy,
            context_recall,
        ],
    )
    return result
