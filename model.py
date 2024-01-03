from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_embedding(input_string):
    """tokenizes string and runs longformer-base-4096 to obtain embedding.
    Currently: Allenai/longformer-base-4096
    Input: String
    Output: Embedding vector, as np.array
    """

    # Load pre-trained BERT tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('allenai/longformer-base-4096')
    model = AutoModel.from_pretrained("allenai/longformer-base-4096")

    # Tokenize input text
    tokens = tokenizer(input_string, return_tensors='pt')

    # Forward pass to obtain embeddings
    with torch.no_grad():
        outputs = model(**tokens)

    # Extract embeddings from the last hidden layer (CLS token)
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()

    return embeddings[0]

def cosine_similarity(embedding1, embedding2):
    """Calculates cosine similarity between two embeddings.
    Input: Two embeddings as np.arrays
    Output: Similarity score between the two embeddings, range: [0,1]
    """

    # Ensure both embeddings have the same shape
    if embedding1.shape != embedding2.shape:
        raise ValueError("Embeddings must have the same shape for cosine similarity calculation.")

    # Calculate cosine similarity
    similarity = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

    return similarity


def get_relevant_papers(question, df, num_papers=10):
    """Finds papers similar to question by calculating cosine similarity of paper embeddings and question embeddings.
    Input:
    - question: string
    - df: pd.DataFrame with column "Embedding"
    - num_papers: Number of papers which should be returned (ranked by similarity in descending order)

    Output:
    - ids_match: indices of papers with the highest similarity score
    - relevant_papers: joined string of the titles and abstracts of the num_papers similar papers
    """

    embedding_question = get_embedding(question)
    similarities = df['Embedding'].apply(lambda embedding_paper: cosine_similarity(embedding_question, embedding_paper))
    ids_match = similarities.sort_values(ascending=False).head(10).index
    relevant_papers = "\n".join(df['TitleAbstract'][ids_match].values)

    return ids_match, relevant_papers


def build_prompt(question, relevant_papers):
    """Builds prompt from a qeustion and relevant papers (concatenated to a single string)
    Note: Chat performance is very sensitive on prompt format. This is simply the result of a few tries.
    """

    prompt = f"""Context information is below.\n---------------\nContext:\n{relevant_papers}\n---------------\n'
    Given the context information and prior knowledge, answer the following query.\n 
    Query: {question}"""

    return prompt