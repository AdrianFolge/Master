import PyPDF2
import textract
import pandas as pd
from pathlib import Path
import re
import uuid
from llama_index.llms.openai import OpenAI
from tqdm.notebook import tqdm
import os
os.environ["OPENAI_API_KEY"] = "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"

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.

    Args:
    pdf_path (str): Path to the PDF file.

    Returns:
    list: List of text chunks extracted from the PDF.
    """
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        # Initialize an empty list to store text chunks
        text_chunks = []
        # Iterate through each page in the PDF
        for page_num in range(len(pdf_reader.pages)):
            # Get the page object
            page = pdf_reader.pages[page_num]
            # Extract text from the page and append it to the list
            text_chunks.append(page.extract_text())
    return text_chunks

def extract_text_from_folder(folder_path):
    """
    Extract text from all PDF files in a folder.

    Args:
    folder_path (str): Path to the folder containing PDF files.

    Returns:
    dict: A dictionary where keys are file names and values are extracted text.
    """
    # Initialize an empty dictionary to store text from each file
    text_dict = {}
    # Iterate through each file in the folder
    for file_path in Path(folder_path).iterdir():
        if file_path.suffix.lower() == '.pdf':
            # Extract text from the PDF file
            text = extract_text_from_pdf(str(file_path))
            # Store the extracted text in the dictionary
            text_dict[file_path.name] = text
    return text_dict

folder_path = './pdf'
pdf_text_dict = extract_text_from_folder(folder_path)

def generate_queries(
    corpus,
    num_questions_per_chunk=1,
    prompt_template=None,
    verbose=False,
):
    """
    Automatisk generer hypotetiske spørsmål som kunne besvares med dokumentet i korpuset.
    """
    llm = OpenAI(model='gpt-3.5-turbo')

    prompt_template = prompt_template or """\
    Kontekstinformasjonen er nedenfor.

    ---------------------
    {context_str}
    ---------------------

    Gitt kontekstinformasjonen og ikke tidligere kunnskap.
    Generer bare spørsmål basert på forespørselen nedenfor.

    Du er en lærer/professor. 
    Oppgaven din er å sette opp {num_questions_per_chunk} spørsmål for en kommende quiz/eksamen. 
    Spørsmålene bør være varierte i naturen på tvers av dokumentet. 
    Begrens spørsmålene til den kontekstinformasjonen som er gitt."
    """    
    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(corpus.items()):
        for chunk in text:
            query = prompt_template.format(context_str=chunk, num_questions_per_chunk=num_questions_per_chunk)
            response = llm.complete(query)

            result = str(response).strip().split("\n")
            questions = [
                re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
            ]
            questions = [question for question in questions if len(question) > 0]

            for question in questions:
                question_id = str(uuid.uuid4())
                queries[question_id] = question
                relevant_docs[question_id] = [node_id]
    return queries, relevant_docs

train_queries, train_relevant_docs = generate_queries(pdf_text_dict)

train_dataset = {
    'Question': train_queries,
    'Corpus': pdf_text_dict,
    'Abstract': train_relevant_docs,
}

dataset = train_dataset

corpus = dataset['Corpus']
queries = dataset['Question']
relevant_docs = dataset['Abstract']

examples = []
for query_id, query in queries.items():
    node_id = relevant_docs[query_id][0]
    text = corpus[node_id]
    example = {"Question" : query, "Abstract" : text}
    examples.append(example)


question_abstract_pair_df = pd.DataFrame(examples)
question_abstract_pair_df.to_csv("./question_abstract_pair.csv")

def generate_answer(
    query,
    context,
    prompt_template=None,
    verbose=False,
):
    """
    Automatisk generer hypotetiske spørsmål som kunne besvares med dokumentasjonen i korpuset.  
    """
    llm = OpenAI(model='gpt-3.5-turbo')

    prompt_template = prompt_template or """\
    
    Kontekstinformasjonen er nedenfor.

    ---------------------
    {context_str}
    ---------------------

    Gitt kontekstinformasjonen og ikke tidligere kunnskap, generer bare svar basert på den nedenfor gitte spørringen.

    ---------------------
    {query_str}
    ---------------------

    Du er en lærer/professor. Oppgaven din er å svare på spørsmål til en kommende quiz/eksamen. Begrens svarene dine basert på den gitte kontekstinformasjonen. Hvis du ikke vet svaret, svar bare: "Jeg vet ikke".
    """
    full_query = prompt_template.format(context_str=context, query_str=query)
    response = llm.complete(full_query)

    result = str(response).strip().split("\n")
    answers = [
            re.sub(r"^\d+[\).\s]", "", answer).strip() for answer in result
        ]
    answers = [answer for answer in answers if len(answer) > 0]
    return answers[0]

for example in tqdm(examples[:100]):
  example["Answer"] = generate_answer(example["Question"], example["Abstract"])

train_df = pd.DataFrame(examples[:52])
train_df.to_csv("question_with_answers.csv")  

def create_synthetic_datasets(synthetic_dataset_path):
    for example in tqdm(examples[:100]):
        example["Answer"] = generate_answer(example["Question"], example["Abstract"])

    train_df = pd.DataFrame(examples[:52])
    train_df.to_csv(synthetic_dataset_path) 