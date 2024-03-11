from RAG import create_RAG
#from create_synthetic_datasets import create_synthetic_datasets
#from evaluate_results import evaluate_predictions

embeddings_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
directory_name = "Master/data/pdfs"
chunk_size = 500
chunk_overlap = 70
dataset_name = "/Users/adrianfolge/Documents/lokal:skole/Master/data/synthetic_data/question_with_answers.csv"
tokenizer_model = "RuterNorway/Llama-2-13b-chat-norwegian"
lm_model = "RuterNorway/Llama-2-13b-chat-norwegian"
output_file_path = "/Users/adrianfolge/Documents/lokal:skole/Master/data/Results"

create_RAG(embeddings_model, directory_name, chunk_size, chunk_overlap, dataset_name, tokenizer_model, lm_model, output_file_path)