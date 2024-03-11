import evaluate
from evaluate import load
import tensorflow_hub as hub
from scipy.spatial import distance
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import pandas as pd

def embed(input, model):
    return model(input)

def SAS(preds, refs, model):
    similarities = []
    embeddings_preds = model.encode(preds)
    embeddings_refs = model.encode(refs)
    for i in range(len(embeddings_preds)):
        similarity = util.pytorch_cos_sim(embeddings_preds[i], embeddings_refs[i])
        similarities.append(similarity[0][0].item())
    average_similarity_score = sum(similarities) / len(similarities)
    return average_similarity_score

references = load_dataset('csv', data_files=r'/Users/adrianfolge/Documents/lokal:skole/Master/data/synthetic_data/question_with_answers.csv', split="train[:10]")
predictions = load_dataset('csv', data_files=r'/Users/adrianfolge/Documents/lokal:skole/Master/data/Results/Faiss_answers_from_model.csv')

def evaluate_predictions(references, predictions, output_file_path):

    bertscore = load("bertscore")
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')

    references = references["Answer"]
    predictions = predictions["train"]["Text"]

    bert_score = bertscore.compute(predictions=predictions, references=references, lang="nb")
    bleu_score = bleu.compute(predictions=predictions, references=references, max_order=2)
    rouge_score = rouge.compute(predictions=predictions, references=references)

    avg_precision = sum(bert_score['precision']) / len(bert_score['precision'])
    avg_recall = sum(bert_score['recall']) / len(bert_score['recall'])
    avg_f1 = sum(bert_score['f1']) / len(bert_score['f1'])

    ## SAS encoder score
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    encoder_model = hub.load(module_url)
    
    list_of_similarity_scores = []
    for i in range(len(predictions)):
        similarity_score = 1-distance.cosine(embed([predictions[i]], encoder_model)[0, :],embed([references[i]], encoder_model)[0, :])
        list_of_similarity_scores.append(similarity_score)
        print(f'\nPrediction: {predictions[i]}\nReference: {references[i]}\nSimilarity Score = {similarity_score} ')
    average_score = sum(list_of_similarity_scores) / len(list_of_similarity_scores)
    print("Average similarity score:", average_score)

    ## SAS transformer score
    transformer_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')



    print("BLEU SCORES")
    print(bleu_score)
    print("ROUGE SCORES")
    print(rouge_score)
    print("BERT SCORES")
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1 Score:", avg_f1)
    print("Average SAS encoder Score:", average_score)
    print("Average SAS transformer Score:", SAS(predictions, references, transformer_model))

    data = {
        "Metric": ["BLEU Score", "ROUGE Score", "Average Precision", "Average Recall", "Average F1 Score", "Average SAS encoder Score", "Average SAS transformer Score"],
        "Score": [bleu_score, rouge_score, avg_precision, avg_recall, avg_f1, average_score, SAS(predictions, references)]
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Specify the file path
    file_path = output_file_path

    # Write DataFrame to CSV
    df.to_csv(file_path, index=False)

    print("Data has been written to", file_path)