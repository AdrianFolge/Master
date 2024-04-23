from datasets import load_dataset
from deep_translator import GoogleTranslator
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def semantic_model(instances, file_path, databases):
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
    list_of_answers = []
    for i in range(instances):
        question = references["spørsmål"][i]
        kommunenavn = references["kommunenavn"][i]
        db = databases[kommunenavn]
        semantic_chunk_retriever = db.as_retriever(search_kwargs={"k" : 1})
        rag_template = """\
        Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.

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
        answer = semantic_rag_chain.invoke(question)
        translated_answer = GoogleTranslator(source='en', target='no').translate(text=answer)
        list_of_answers.append(translated_answer)
    return list_of_answers


