import json
import numpy as np
import pandas as pd
import torch
from operator import itemgetter
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

class SubQueryColumnMapper:
    def __init__(self, model_name='facebook/contriever-msmarco', batch_size=200):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.batch_size = batch_size
     
    def embed(self, texts, hide_progress=False):
        embeds = []
        for i in tqdm(range((len(texts) // self.batch_size) + 1), disable=hide_progress):
            _texts = texts[i * self.batch_size:(i + 1) * self.batch_size]
            if len(_texts) == 0:
                break
            
            inputs = self.tokenizer(_texts, padding=True, truncation=True, return_tensors='pt')
            
            with torch.no_grad():
                vec = self.model(**inputs)
            
            vec = mean_pooling(vec[0], inputs['attention_mask'])
            embeds.append(vec.cpu().numpy())
        
        if len(embeds) == 0:
            return np.array([])
        
        embeds = np.vstack(embeds)
        return embeds
     
    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
       
    def map_subQueries_to_columns(self, sub_queries, tables, corpus_tables, questions):
        mapping = []
        for idx, (sqs, tbls) in enumerate(zip(sub_queries, tables)):
            sq_map = []
            columns = []
            for tbl_key in tbls:
                tbl = corpus_tables[tbl_key]
                for col in tbl["column_names"]:
                    columns.append({
                        "table": tbl_key,
                        "column": col,
                        "table_name": tbl['table_name'],
                        "column_name": col,
                        "column_key": f"{tbl['table_name']} {col}"
                    })
           
            if not columns:
                mapping.append({
                    "query": questions[idx],
                    "sub_queries_mapping": []
                })
                continue
           
            column_texts = [c["column_key"] for c in columns]
            column_embeds = self.embed(column_texts, hide_progress=True)
           
            for sq in sqs:
                sq_cleaned = sq.replace(':', ' ')
                sq_vector = self.embed([sq_cleaned], hide_progress=True)[0]
               
                sims = [self.cosine_similarity(sq_vector, col_embed) for col_embed in column_embeds]
                   
                best_matches = [
                    {"table": c["table"], "column": c["column"], "score": float(s)}
                    for c, s in zip(columns, sims)
                ]
                best_matches.sort(key=itemgetter("score"), reverse=True)
                sq_map.append({
                    "sub_query": sq,
                    "top_matches": best_matches[:2]
                })
           
            query_with_subqueries = {
                "query": questions[idx],
                "sub_queries_mapping": sq_map
            }
            mapping.append(query_with_subqueries)
        return mapping
                   
if __name__ == "__main__":
    with open("jar2-main/data/bird/decomp.json") as f:
        sub_queries_groups = json.load(f)
    with open("jar2-main/data/ilp_preds/contriever/bird_0_expanded.json") as f:
        expanded_tables_groups = json.load(f)
    with open("jar2-main/data/bird/dev_tables.json") as f:
        dev_tables = json.load(f)
   
    questions_df = pd.read_json("jar2-main/data/bird/dev.json")
    questions_list = questions_df["question"].tolist()
   
    mapper = SubQueryColumnMapper()
    mapping = mapper.map_subQueries_to_columns(
        sub_queries_groups, 
        expanded_tables_groups, 
        dev_tables, 
        questions_list
    )
   
    print("done")
   
    with open("jar2-main/data/ilp_preds/contriever/subquery_to_column_mapping_semantic.json", "w") as f:
        json.dump(mapping, f, indent=2)
