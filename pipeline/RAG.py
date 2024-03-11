from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os
from langchain_community.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from datasets import load_dataset
import pandas as pd

def create_RAG(embeddings_model, directory_name, chunk_size, chunk_overlap, dataset_name, tokenizer_model, lm_model, output_file_path):
    loader = DirectoryLoader(directory_name, glob="**/*.pdf", show_progress=True, loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name=embeddings_model)

    db = Chroma.from_documents(docs, embeddings)
    dataset = load_dataset('csv', data_files=dataset_name, split="train[:10]")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    model = AutoModelForCausalLM.from_pretrained(lm_model)

    answers_from_model = []
    for i in range(10):
        query = dataset["Question"][i]
        found_docs = db.similarity_search(query)
        context = found_docs[0].page_content
        input = f"Spørsmål: {query} context: {context}"
        instruction = "Svar på spørsmålet basert på det som står i 'context'"
        prompt_template=f'''### Instruction: {instruction}
        ### Input: {input}
        ### Response:
        '''
        print("\n\n*** Generate:")
        inputs = tokenizer(prompt_template, return_tensors="pt")

        out = model.generate(**inputs, max_new_tokens=200)
        print(tokenizer.decode(out[0], skip_special_tokens=True))

        # Pipeline prompting
        print("\n\n*** Pipeline:\n\n")
        pipe = pipeline(
            "text-generation",
            model=model,
            do_sample=True,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        print(pipe(prompt_template)[0]['generated_text'][len(prompt_template):])
        answers_from_model.append(pipe(prompt_template)[0]['generated_text'][len(prompt_template):])

    # Convert the list to a pandas DataFrame
    df = pd.DataFrame(answers_from_model, columns=['Text'])

    # Specify the file path
    file_path = output_file_path

    # Write the DataFrame to a CSV file
    df.to_csv(file_path, index=False)




