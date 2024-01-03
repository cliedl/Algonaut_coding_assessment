import pandas as pd
import numpy as np
from openai import OpenAI

from model import get_embedding, cosine_similarity, get_relevant_papers, build_prompt

# Insert API key here:
api_key = None


# Load dataset with arXiv papers on Llama-2
df = pd.read_csv("df_papers_llama2.csv")
Embeddings = np.load("Embeddings.npy", allow_pickle=True)
df["Embedding"] = Embeddings

if __name__ == "__main__":

    # Input api_key if not available
    if api_key == None:
        print('Please input API key')
        api_key = input(str)

    # Start client
    client = OpenAI(api_key=api_key)
    
    while True:
        print('________________________________________________________')
        print('What is your question?')
        question = input(str)

        # Get relevant papers
        paper_ids, relevant_papers = get_relevant_papers(question, df, num_papers=10)
        
        # Construct prompt from question and relevant papers
        prompt = build_prompt(question, relevant_papers)

        # Query GPT-3.5-turbo
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Print response
        print(response.choices[0].message.content)
