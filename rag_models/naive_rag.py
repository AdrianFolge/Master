from deep_translator import GoogleTranslator
from datasets import load_dataset

def naive_rag_translated(instances, file_path, databases):
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]")
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

def naive_rag(instances, file_path, databases):
    references = load_dataset('csv', data_files={file_path}, split=f"train[:{instances}]") 
    list_of_answers = []
    for num in range(instances):
        query = references["spørsmål"][num]
        kommunenavn = references["kommunenavn"][num]
        db = databases[kommunenavn]
        found_docs = db.similarity_search(query)
        context = found_docs[0].page_content
        list_of_answers.append(context)
    return list_of_answers