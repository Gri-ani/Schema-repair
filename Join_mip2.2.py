from mip import *
import numpy as np
from tqdm import tqdm
import argparse
import json
import pickle
import os
from groq import Groq
 
client = Groq(api_key="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")

def get_skip_idxs(dataset: str):
  qs = read_json(f'./jar2-main/data/{dataset}/dev.json')  # Fix path
  skip_idxs = [i for i in range(len(qs)) if i not in (0,1,2,3,4)]
  return skip_idxs

# Get join relevance from join_matrix
def get_join_score(table_i, table_j, query_idx):
    join_matrix = read_json("./join_predictions_bird66.json")
    key = f"{table_i}-{table_j}"
    if key not in join_matrix:
        return 0
    score = join_matrix[key].get(str(query_idx), 0)
    if isinstance(score, str):
        return 0 if score == False else 0.5
    return score

def get_corpus(dataset):
  tables = read_json(f'./jar2-main/data/{dataset}/dev_tables.json')
  return list(tables.keys())

def read_json(fn):
  with open(fn) as f:
    return json.load(f)
  
def write_json(obj, fn):
  with open(fn, 'w') as f:
    json.dump(obj, f, indent=2)

def read_pickle(fn):
  with open(fn, 'rb') as f:
    return pickle.load(f)

def write_pickle(r, fn):
  with open(fn, 'wb') as f:
    pickle.dump(r, f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_edges_file(file_path):
    with open(file_path, 'r') as f:
        edges_data = json.load(f)
    return edges_data

# def assign_columns(lm, q_scores, dq_scores, dataset, t_num, w, num_partitions, partition, fn, join_matrix):
#     w1, w2, w3 = w
#     corpus_tables = get_corpus(dataset)
#     qs = read_json(f'./jar2-main/data/{dataset}/dev.json')
#     tables = read_json(f'./jar2-main/data/{dataset}/dev_tables.json')
#     interval = len(qs) // num_partitions + 1
#     start_idx, end_idx = partition * interval, (partition + 1) * interval
#     fn = f'{fn}_{partition}.json'
#     # skip_idxs = get_skip_idxs(dataset)
#     skip_idxs=[]
#     preds = []
#     for q_idx, q in enumerate(tqdm(qs)):
#         if not (start_idx <= q_idx < end_idx):
#             continue 
#         # if q_idx in skip_idxs:
#         #     preds.append([])
#         #     continue
#         if q_idx >= 5:
#             preds.append([])
#             continue
#         print(q_idx)
#         m = Model(sense=MAXIMIZE)  
#         score = q_scores[q_idx]

#         top_idxs = np.argsort(-score)[:t_num]
#         top_tables = [corpus_tables[top_idx] for top_idx in top_idxs]
#         print(f"Length of corpus_tables: {len(corpus_tables)}")
#         print(f"Top indices predicted: {top_idxs}")
#         num_q = len(dq_scores[q_idx])
#         k_num = num_q+1
#         join_num = k_num   
#         # dq_scores - sub-query to column relevance 
#         tr = [[] for _ in range(num_q)]
#         for dq_idx, dq_score in enumerate(dq_scores[q_idx]):
#             for t in top_tables:
#                 tr[dq_idx].append(dq_score[corpus_tables.index(t)])
#         # normalize scores
#         if lm == 'tapas' and dataset == 'spider':
#             score_max, score_min = np.amax(score), np.amin(score)
#             if score_max > score_min:
#                 score = (score - score_min) / (score_max - score_min)  
#         num_tables = len(top_tables)
#         num_cols = [len(tables[t]['column_names_original']) for t in top_tables]
#         # Decision variables
#         T = [m.add_var(var_type=BINARY, name=f'T{i}') for i in range(num_tables)]

        
#         C = [[[m.add_var(var_type=BINARY, name=f'C_{q}_{i}_{k}') 
#                for k in range(num_cols[i])] 
#               for i in range(num_tables)] 
#              for q in range(num_q)]
        
#         J = [[m.add_var(var_type=BINARY, name=f'J_{i}_{j}')
#               for j in range(num_tables)] 
#              for i in range(num_tables)]
        
#         # Constraints
#         # Constraint 1: number of tables selected equals the number of sub-queries
#         m += xsum(T[i] for i in range(num_tables)) <= num_q + 1
        
#         # Constraint 2: each sub-query covered by at most one column per table
#         for q in range(num_q):
#             for i in range(num_tables):
#                 m += xsum(C[q][i][k] for k in range(num_cols[i])) <= 1
        
#         # Constraint 3: can only map to columns of selected tables
#         for q in range(num_q):
#             for i in range(num_tables):
#                 for k in range(num_cols[i]):
#                     m += C[q][i][k] <= T[i]
        
#         # Constraint 4: J[i][j] is 1 only if both tables i and j are selected
#         for i in range(num_tables):
#             for j in range(num_tables):
#                 if i != j:
#                     m += J[i][j] <= T[i]
#                     m += J[i][j] <= T[j]
#                     m += J[i][j] >= T[i] + T[j] - 1
        
#         # Optional Constraint 5: each sub-query must be covered by at least one column
#         for q in range(num_q):
#             m += xsum(C[q][i][k] for i in range(num_tables) for k in range(num_cols[i])) >= 1
        
#         # Objective Function
#         # O1: Table-Query Relevance
#         o1 = w1 * xsum(score[corpus_tables.index(top_tables[i])] * T[i] 
#                        for i in range(num_tables))
        
#         # O2: Sub-query column relevance
#         o2 = w2 * xsum(tr[q][i][k] * C[q][i][k] 
#                        for q in range(num_q) 
#                        for i in range(num_tables) 
#                        for k in range(num_cols[i]))
        
#         # O3: Table-table joinability with sub-query coverage
#         cover = [m.add_var(var_type=INTEGER, name=f'cover_{i}') 
#                  for i in range(num_tables)]
        
#         for i in range(num_tables):
#             m += cover[i] == xsum(C[q][i][k] 
#                                   for q in range(num_q) 
#                                   for k in range(num_cols[i]))
#         M = 100
#         aux = [[m.add_var(var_type=INTEGER, lb=0, ub=M, name=f'aux_{i}_{j}') for j in range(num_tables)] for i in range(num_tables)]
#         # Get join relevance from join_matrix

#         for i in range(num_tables):
#             for j in range(num_tables):
#                 if i!=j:
#                     cover_sum = cover[i] + cover[j]
#                     m+= aux[i][j] <= M*J[i][j]
#                     m+= aux[i][j] <=cover_sum
#                     m+= aux[i][j] >= cover_sum - M * (1-J[i][j])
#                     m+= aux[i][j] >= 0
#         o3 = w3 * xsum ((1 + get_join_score(top_tables[i], top_tables[j], q_idx)) * aux[i][j]
#                for i in range(num_tables) 
#                for j in range(num_tables) 
#                if i != j)
        
#         obj = o1 + o2 + o3
#         m.objective = obj
#         m.verbose = 0
        
#         status = m.optimize(max_seconds=60)
        
#         if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
#             r_idxs = [int(v.name[1:]) for v in m.vars 
#                       if abs(v.x) > 1e-6 and v.name.startswith('T') and '_' not in v.name]
#             r_tables = [top_tables[r_idx] for r_idx in r_idxs]
            
#             print(f"Query {q_idx}: Selected tables: {r_tables}")
#             for i in r_idxs:
#                 covered_subqueries = []
#                 for q in range(num_q):
#                     for k in range(num_cols[i]):
#                         if C[q][i][k].x > 0.5:
#                             covered_subqueries.append(q)
#                             break
#                 print(f"  Table {top_tables[i]} covers sub-queries: {covered_subqueries}")
            
#             preds.append(r_tables)
#         else:
#             print(f"Optimization failed for query {q_idx}")
#             preds.append([])
        
#         write_json(preds, fn)
    
#     write_json(preds, fn)
#     return preds
from mip import *
import numpy as np
from tqdm import tqdm

def assign_columns(lm, q_scores, dq_scores, dataset, t_num, w, num_partitions, partition, fn, join_matrix):
    w1, w2, w3 = w
    corpus_tables = get_corpus(dataset)
    qs = read_json(f'./jar2-main/data/{dataset}/dev.json')
    tables = read_json(f'./jar2-main/data/{dataset}/dev_tables.json')
    interval = len(qs) // num_partitions + 1
    start_idx, end_idx = partition * interval, (partition + 1) * interval
    fn = f'{fn}_{partition}.json'
    skip_idxs = []
    preds = []

    for q_idx, q_text in enumerate(tqdm(qs)):
        if not (start_idx <= q_idx < end_idx):
            continue
        # if q_idx in skip_idxs or q_idx >= 5:
        #     preds.append([])
        #     continue

        print(f"Processing query {q_idx}")
        m = Model(sense=MAXIMIZE)
        score = q_scores[q_idx]

        top_idxs = np.argsort(-score)[:t_num]
        top_tables = [corpus_tables[top_idx] for top_idx in top_idxs]
        num_q = len(dq_scores[q_idx])
        num_tables = len(top_tables)
        num_cols = [len(tables[t]['column_names_original']) for t in top_tables]

        # Normalize score for tapas/spider
        if lm == 'tapas' and dataset == 'spider':
            s_max, s_min = np.amax(score), np.amin(score)
            if s_max > s_min:
                score = (score - s_min) / (s_max - s_min)

        # ----------------------------
        # Decision variables
        # ----------------------------
        T = [m.add_var(var_type=BINARY, name=f'T{i}') for i in range(num_tables)]

        # C[q][i][k] = 1 if sub-query q maps to column k of table i
        C = [[[m.add_var(var_type=BINARY, name=f'C_{dq}_{i}_{k}') 
               for k in range(num_cols[i])] 
              for i in range(num_tables)] 
             for dq in range(num_q)]

        # J[i][j] = 1 if tables i and j are both selected
        J = [[m.add_var(var_type=BINARY, name=f'J_{i}_{j}') for j in range(num_tables)] for i in range(num_tables)]

        # ----------------------------
        # Constraint: all selected tables from same DB
        # ----------------------------
        db_ids = [tables[t]['db_id'] for t in top_tables]
        unique_dbs = list(dict.fromkeys(db_ids))
        db_to_idx = {db: idx for idx, db in enumerate(unique_dbs)}
        D = [m.add_var(var_type=BINARY, name=f'D_{d}') for d in range(len(unique_dbs))]

        for i, db in enumerate(db_ids):
            db_idx = db_to_idx[db]
            m += T[i] <= D[db_idx]

        m += xsum(D[d] for d in range(len(unique_dbs))) <= 1  # at most one DB

        # ----------------------------
        # Constraint: each sub-query maps to exactly one table
        # ----------------------------
        U = [[m.add_var(var_type=BINARY, name=f'U_{dq}_{i}') for i in range(num_tables)] for dq in range(num_q)]
        for dq in range(num_q):
            for i in range(num_tables):
                # link U and C
                m += xsum(C[dq][i][k] for k in range(num_cols[i])) <= U[dq][i] * num_cols[i]
                m += U[dq][i] <= xsum(C[dq][i][k] for k in range(num_cols[i]))
            m += xsum(U[dq][i] for i in range(num_tables)) == 1

        # ----------------------------
        # Other constraints
        # ----------------------------

        # Table count constraint: at most num_q
        m += xsum(T[i] for i in range(num_tables)) <= num_q

        # Each sub-query covered by at most one column per table
        for dq in range(num_q):
            for i in range(num_tables):
                m += xsum(C[dq][i][k] for k in range(num_cols[i])) <= 1

        # Column can only map if table selected
        for dq in range(num_q):
            for i in range(num_tables):
                for k in range(num_cols[i]):
                    m += C[dq][i][k] <= T[i]

        # J[i][j] linkage
        for i in range(num_tables):
            for j in range(num_tables):
                if i != j:
                    m += J[i][j] <= T[i]
                    m += J[i][j] <= T[j]
                    m += J[i][j] >= T[i] + T[j] - 1

        # Each sub-query must be covered by at least one column
        for dq in range(num_q):
            m += xsum(C[dq][i][k] for i in range(num_tables) for k in range(num_cols[i])) >= 1

        # ----------------------------
        # Objective
        # ----------------------------
        # O1: Table-Query Relevance
        o1 = w1 * xsum(score[corpus_tables.index(top_tables[i])] * T[i] for i in range(num_tables))

        # O2: Sub-query column relevance
        tr = [[] for _ in range(num_q)]
        for dq_idx, dq_score in enumerate(dq_scores[q_idx]):
            for t in top_tables:
                tr[dq_idx].append(dq_score[corpus_tables.index(t)])
        o2 = w2 * xsum(tr[dq][i][k] * C[dq][i][k] for dq in range(num_q) for i in range(num_tables) for k in range(num_cols[i]))

        # O3: Table-table joinability
        cover = [m.add_var(var_type=INTEGER, name=f'cover_{i}') for i in range(num_tables)]
        for i in range(num_tables):
            m += cover[i] == xsum(C[dq][i][k] for dq in range(num_q) for k in range(num_cols[i]))

        M = 100
        aux = [[m.add_var(var_type=INTEGER, lb=0, ub=M, name=f'aux_{i}_{j}') for j in range(num_tables)] for i in range(num_tables)]
        for i in range(num_tables):
            for j in range(num_tables):
                if i != j:
                    cover_sum = cover[i] + cover[j]
                    m += aux[i][j] <= M * J[i][j]
                    m += aux[i][j] <= cover_sum
                    m += aux[i][j] >= cover_sum - M * (1 - J[i][j])
                    m += aux[i][j] >= 0

        o3 = w3 * xsum((1 + get_join_score(top_tables[i], top_tables[j], q_idx)) * aux[i][j]
                       for i in range(num_tables) for j in range(num_tables) if i != j)

        m.objective = o1 + o2 + o3
        m.verbose = 0

        status = m.optimize(max_seconds=60)

        if status in [OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE]:
            r_idxs = [int(v.name[1:]) for v in m.vars if v.name.startswith('T') and '_' not in v.name and abs(v.x) > 1e-6]
            r_tables = [top_tables[i] for i in r_idxs]
            print(f"Query {q_idx}: Selected tables: {r_tables}")
            preds.append(r_tables)
        else:
            print(f"Optimization failed for query {q_idx}")
            preds.append([])

        write_json(preds, fn)

    write_json(preds, fn)
    return preds


if __name__ == '__main__':

    partition = 0      
    num_partitions = 1  
    dataset = 'bird'    
    model = 'contriever' 


    q_scores = np.load(f'./jar2-main/data/{dataset}/{model}/score.npy')
    print(f"Query scores shape: {q_scores.shape}")

    dq = read_json(f'./jar2-main/data/{dataset}/decomp.json')
    dq_scores = read_pickle(f'./jar2-main/data/{dataset}/contriever/score_decomp.pkl')


    q_interval = [0]
    for q in dq:
        q_interval.append(q_interval[-1] + len(q))
    dq_scores = [dq_scores[q_interval[i]:q_interval[i+1]] for i in range(len(dq))]


    t_num = 10
    if dataset == 'bird':
        w = [3, 8, 1]
    elif dataset == 'spider':
        w = [2, 8, 1]

    fn = f'./jar2-main/data/ilp_preds/{model}/{dataset}'


    join_matrix = read_json("./join_predictions_bird66.json")


    print(f"Running partition {partition}/{num_partitions}")
    preds = assign_columns(
        model, q_scores, dq_scores, dataset, t_num, w,
        num_partitions, partition, fn, join_matrix
    )

    print(preds)

    print(f"Partition {partition} complete. Results saved to {fn}_{partition}.json")
    preds_path = f'./jar2-main/data/ilp_preds/{model}/{dataset}_{partition}.json'
    preds = read_json(preds_path)
    qa_pairs = [
        {
            "question": dq[i],
            "predicted_answer": preds[i]
        }
        for i in range(len(preds))
    ]

    # # Save combined results
    output_path = f'{fn}_{partition}_qa.json'
    write_json(qa_pairs, output_path)

    print(f"Partition {partition} complete. Results (Q&A) saved to {output_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-p', '--partition', type=int, required=True)
#     parser.add_argument('-d', '--dataset', type=str, default='bird', choices=['bird', 'spider'])
#     parser.add_argument('-m', '--model', type=str, default='contriever', choices=['contriever', 'tapas'])
#     parser.add_argument('--build-join-matrix', action='store_true', help='Build join relevance matrix')
#     args = parser.parse_args()
#     num_partitions = 1
#     dataset = args.dataset
#     model = args.model
#     q_scores = np.load(f'./jar2-main/data/{dataset}/{model}/score.npy')
#     print(f"Query scores shape: {q_scores.shape}")
#     dq = read_json(f'./jar2-main/data/{dataset}/decomp.json')
#     dq_scores = read_pickle(f'./jar2-main/data/{dataset}/contriever/score_decomp.pkl')
    
#     q_interval = [0]
#     for q in dq:
#         q_interval.append(q_interval[-1] + len(q))
#     dq_scores = [dq_scores[q_interval[i]:q_interval[i+1]] for i in range(len(dq))]
#     t_num = 10
#     if dataset == 'bird':
#         if model == 'contriever':
#             w = [3, 8, 1]
#         elif model == 'tapas':
#             w = [3, 8, 1]
#     elif dataset == 'spider':
#         w = [2, 8, 1]
    
#     fn = f'./jar2-main/data/ilp_preds/{model}/{dataset}'
#     # if args.build_join_matrix:
#     #     print("Building join relevance matrix...")
#     #     edges_data = load_edges_file(f'./jar2-main/data/{dataset}/join_edges2.json')
#     #     corpus_tables = get_corpus(dataset)
#     #     join_cache_file = f'./jar2-main/data/{dataset}/join_relevance_cache'
#     #     join_matrix = join_dict_all_queries(dataset, join_cache_file, edges_data, corpus_tables)
#     #     print(f"Join matrix saved to {join_cache_file}.json")
#     # else:
#     #     join_cache_file = f'./jar2-main/data/{dataset}/join_relevance_cache.json'
#     #     if os.path.exists(join_cache_file):
#     #         join_matrix = load_or_init_cache(join_cache_file)
#     #         print(f"Loaded join matrix from {join_cache_file}")
#     #     else:
#     #         print("Warning: No join matrix found. Using default values (0).")
#     #         join_matrix = {} 
#     join_matrix = read_json(".\join_predictions_bird66.json")
#     print(f"Running partition {args.partition}/{num_partitions}")
#     preds = assign_columns(model, q_scores, dq_scores, dataset, t_num, w, 
#                           num_partitions, args.partition, fn, join_matrix)
#     print(f"Partition {args.partition} complete. Results saved to {fn}_{args.partition}.json")




    # from mip import *
# import numpy as np
# from tqdm import tqdm
# import argparse
# import json
# from utils import read_json, read_pickle, write_json, get_corpus, merge, get_skip_idxs
# from compatibility import get_cr
# # from joinability_graph import get_query_specific_join_matrix
# import os

# def read_json(path):
#   with open(path, 'r') as f:
#     return json.load(f)
  
# def save_json(path, data):
#   with open(path, 'w') as f:
#     json.dump(data, f, indent=2)

# def load_edges_file(file_path):
#   with open(file_path, 'r') as f:
#     edges_data = json.load(f)

#   return edges_data

# def get_table_schema(tables_data, table_key):
#     if table_key in tables_data:
#         table_info = tables_data[table_key]
#         schema_info = {
#             "table_name": table_info.get("table_name", ""),
#             "columns": table_info.get("column_names_original", [])
#         }
#         return schema_info
#     return None

# def check_join_exists(table1, table2, edges_data):
#   edge_key1 = f"{table1}-{table2}"
#   edge_key2 = f"{table2}-{table1}"

#   return edge_key1 in edges_data or edge_key2 in edges_data

# def load_or_init_cache(path):
#   if os.path.exists(path):
#     with open(path, 'r') as f:
#       return json.load(f)
#     return {}

# def get_join_relevance_llm( table1, table2, edges_data, query_text ):
#   tables = read_json(f'./data/{dataset}/dev_tables.json')
#   # LLM call to determine if the join between table1 and table2 is relevant for the query 
#   # returns 1 if relevant 0 otherwise
#   join_key1 = f"{table1}#sep#{table2}"
#   join_key2 = f"{table2}#sep#{table1}"
#   if join_key1 in edges_data:
#     join_key = join_key1
#     possible_joins = edges_data[join_key1]
#   elif join_key2 in edges_data:
#     join_key = join_key2
#     possible_joins = edges_data[join_key2]
#   else:
#     return 0 
#   print(f"LLM call for join between {table1} and {table2} for query: {query_text}")
#   prompt = f"""
#   Query: "{query_text}"
#   Table Schema for {table1.split('#sep#')[-1]}:
#   """

#   schema1 = get_table_schema(tables, table1)
#   if schema1:
#     prompt += f"Table: {schema1["table_name"]}\n"
#     prompt += f"Columns: {', '.join(schema1["columns"])}\n\n"
  
#   prompt += f"Table Schema for {table2.split('#sep#')[-1]}:\n"
  
#   schema2 = get_table_schema(tables, table2)
#   if schema1:
#     prompt += f"Table: {schema2["table_name"]}\n"
#     prompt += f"Columns: {', '.join(schema2["columns"])}\n\n"
  
#   prompt += f"Possible joins between tables {table1} and {table2}:\n"
#   prompt += json.dumps(possible_joins, indent=2)

#   prompt += f"""
#   The information shared above shows the joins possible between the tables {table1} and {table2}. You have to tell based on the query, if any of the joins given above is a valid join for the query or not

#   Consider:
#   1. Wether any of the provided join columns is actually the correct join to answer the query correctly
#   2. Other calculated information has also been shared just for reference

#   You have to respond with a '1' if (one of the joins porovided above is a valid join for this query) else return '0' if none of the join columns between the 2 tables given above is a valid join for the given queryw.
  
  
#   """

#   try:
#     # groq api call
#     llm_response = response.choices[0].message.content.strip()

#     if '1' in llm_response:
#       return 1
#     if '0' in llm_response:
#       return 0
#     else:
#       print("error in llm response")

#   except:
#     print(f"LLM call failed")

  


# def join_dict_per_query(table_keys, edges_data):
#   qs = read_json(f'./data/{dataset}/dev.json')
#   cache_file = f'{fn}.json'
#   join_cache = load_or_init_cache(cache_file)

#   preds = []
#   for q_idx, q in enumerate(tqdm(qs)):
#     query_text = q["question"] 
#     preds_for_Q = {}

#     for table_key1 in table_keys:
#       for table_key2 in table_keys:
#         if table_key1 == table_key2:
#           continue

#         if check_join_exists(table_key1, table_key2, edges_data):
#           join_key = f"{table_key1}-{table_key2}"

#           if join_key not in join_cache:
#             join_cache[join_key] = {}

#           if str(q_idx) not in join_cache[join_key]:
#             join_cache[join_key][str(q_idx)] = get_join_relevance_llm(table_key1, table_key2,edges_data, query_text)
#             save_json(cache_file, join_cache)

#           preds_for_Q[join_key] = join_cache[join_key][str(q_idx)]
    
#     preds.append(preds_for_Q)

#   return preds


# def join_dict_all_queries(dataset, fn, edges_data, tables):
#   qs = read_json(f'./data/{dataset}/dev.json')

#   cache_file = f'{fn}.json'
#   join_cache = load_or_init_cache(cache_file)

#   preds = []
#   for q_idx, q in enumerate(tqdm(qs)):
#     query_text = q["question"] if isinstance(q, dict) else str(q)
#     preds_for_q = {}

#     for table_key1 in tables:
#       for table_key2 in tables:
#         if table_key1 == table_key2:
#           continue
#         if check_join_exists(table_key1, table_key2, edges_data):
#           join_key = f"{table_key1}-{table_key2}"

#           if join_key not in join_cache:
#             join_cache[join_key] = {}

#           if str(q_idx) not in join_cache[join_key]:
#             join_cache[join_key][str(q_idx)] = get_join_relevance_llm(table_key1, table_key2, query_text)

#           preds_for_q[join_key] = join_cache[join_key][str(q_idx)]

#       preds.append(preds_for_q)


#   save_json(cache_file, join_cache)
#   return preds

















# def assign_columns(lm, q_scores, dq_scores, dataset, t_num, w, num_partitions, partition, fn):
#   w1, w2, w3, w4 = w
#   corpus_tables = get_corpus(dataset)
#   qs = read_json(f'./data/{dataset}/dev.json')
#   tables = read_json(f'./data/{dataset}/dev_tables.json')

#   interval = len(qs) // num_partitions + 1
#   start_idx, end_idx = partition * interval, (partition + 1) * interval

#   fn = f'{fn}_{partition}.json'

#   skip_idxs = get_skip_idxs(dataset)
#   preds = []
#   for q_idx, q in enumerate(tqdm(qs)):
#     if not (start_idx <= q_idx < end_idx):
#       continue

#     if q_idx in skip_idxs:
#       preds.append([])
#       continue

#     m = Model(sense=MAXIMIZE)

#     score = q_scores[q_idx]

#     top_idxs = np.argsort(-score)[:t_num]
#     top_tables = [corpus_tables[top_idx] for top_idx in top_idxs]
    
#     num_q = len(dq_scores[q_idx])
#     k_num = num_q
#     join_num = k_num-1

#     # dq_scores - sub-query to column relevance 
#     # all subqueries are flattened --> for each subquery, compute similarity to all columns in all tables
#     tr = [[] for _ in range(num_q)]
#     for dq_idx, dq_score in enumerate(dq_scores[q_idx]):
#       for t in top_tables:
#         tr[dq_idx].append(dq_score[corpus_tables.index(t)])      
    
#     # normalize scores
#     if lm == 'tapas' and dataset == 'spider':
#       score_max, score_min = np.amax(score), np.amin(score)
#       score = (score - score_min) / (score_max - score_min)
    
#     num_tables = len(top_tables)
#     num_cols = [len(tables[t]['column_names_original']) for t in top_tables]


    
#     M = 1000000
# # Ti is decision variable if table i is selected or not
#     T = [m.add_var(var_type = BINARY, name =f'T{i}') for i in range(num_tables)]
      
# # Cquik is decision var for wether sub-query q is mapped to column k of table i
#     C = [[[m.add_var(var_type=BINARY, name=f'C_{q}_{i}_{k}') 
#                for k in range(num_cols[i])] 
#               for i in range(num_tables)] 
#              for q in range(num_q)]
#     m += xsum(b[i] for i in range(num_tables)) == k_num
#     m += xsum(c_ij_kl[i][j][k][l] for i in range(num_tables) for j in range(num_tables) for k in range(num_cols[i]) for l in range(num_cols[j])) <= join_num
# # Jij is also binary decision variable for wether tables i and j are both selected or not (join consideration)

#     J = [[m.add_var(var_type = BINARY, name = f'J_{i}_{j}')for j in range(num_tables)] for i in range (num_tables)]

#     # constraints
#     # constraint 1 - number of tables selected equals the number of sub-queries
#     m += xsum(T[i] for i in range(num_tables)) == num_q

#     # constraint 2 - each sub-query covered by at most one column per table
#     # ** try both with and without this
#     for q in range(num_q):
#       for i in range(num_tables):
#         m+= xsum(C[q][i][k] for k in range(num_cols[i])) <= 1

#     # constraint 3 - can only map to columns of selected tables
#     for q in range(num_q):
#       for i in range(num_tables):
#         m += C[q][i][k] <= T[i]

#     # constraint 4 - J[i][j] it is 1 only if both tables i and j are selected
#     for i in range(num_tables):
#       for j in range(num_tables):
#         m += J[i][j] <= T[i]
#         m += J[i][j] <= T[j]
#         m += J[i][j] >= T[i] + T[j] - 1

#   #  constraint 5 - optional 
#   # for q in range(num_q):
#   #    m += xsum(C[q][i][k] for i in range(num_tables) for k in range(num_cols[i])) >= 1

    

#   # OBJECTIVE FUNCTION

#   # Table- Query Relevance 

#     o1 = w1 * xsum(score[corpus_tables.index(top_tables[i])] * T[i] for i in range(num_tables))

#   # sub-query col relevance 

#     o2 = w2 * xsum(tr[q][i][k] * C[q][i][k] 
#                           for q in range(num_q) 
#                           for i in range(num_tables) 
#                           for k in range(num_cols[i]))
  
#   # table table joinability with sub-query coverage

#     cover = [m.add_var(var_type = INTEGER, name = f'cover_{i}')for i in range(num_tables)]
#   # cover_i is - which is the sub-queries covered by table i
#     for i in range(num_tables):
#       m += cover[i] == xsum(C[q][i][k] for q in range(num_q) for k in range(num_cols[i]))
#   # o3 ∑Jij⋅(1+Join(i,j))⋅(cover_i + cover_j)
#     # join[i][j] - would be NL based  - external function call
#     # join array validate before hand with NL 
#     # join multidimensional 
#     # before hand based on NL join matrix - used later at runtime
#     o3 = w3 * xsum(J[i][j] * (1 + join[i][j][q_idx]) * (cover[i] + cover[j])
#                           for i in range(num_tables) 
#                           for j in range(num_tables) 
#                           if i != j)
#     obj = o1 + o2 + o3
#     m.objective = maximize(obj)
#     m.verbose = 0
#     status = m.optimize(max_seconds=60)
#     if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
#       r_idxs = [int(v.name[1:]) for v in m.vars if abs(v.x) > 1e-6 and 'b' in v.name and '_' not in v.name]
#       r_tables = [top_tables[r_idx] for r_idx in r_idxs]
#       print(r_tables)
#       for i in r_idxs:
#         covered_subqueries = []
#         for q in range(num_q):
#           for k in range(num_cols[i]):
#             if C[q][i][k].x > 0.5:
#               covered_subqueries.append(q)
#         print(f"  Table {top_tables[i]} covers sub-queries: {covered_subqueries}")
#       preds.append(r_tables)
#     else:
#       print("optimization failed")
#       preds.append([]) 
#     write_json(preds, fn) 
#   write_json(preds, fn)

# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('-p', '--partition', type=int)
#   args = parser.parse_args()
#   num_partitions = 40
#   dataset = ['bird', 'spider'][1]
#   model = ['contriever', 'tapas'][0]
#   q_scores = np.load(f'./data/{dataset}/{model}/score.npy')
#   print(q_scores.shape)
#   dq = read_json(f'./data/{dataset}/decomp.json')
#   dq_scores = read_pickle(f'./data/{dataset}/contriever/score_decomp.pkl')
#   q_interval = [0]
#   for q in dq:
#     q_interval.append(q_interval[-1] + len(q))
#   dq_scores = [dq_scores[q_interval[i]:q_interval[i+1]] for i in range(len(dq))]
#   # k_num is the number of tables in the output
#   # t_num is the number of tables provided to the MIP program (from contriever/ tapas)
#   # k_num = 
#   t_num = 10  
#   if dataset == 'bird':
#     if model == 'contriever':
#       w = [3, 8, 1]
#     elif model == 'tapas':
#       w = [3, 8, 1]
#   elif dataset == 'spider':
#     w = [2, 8, 1]
#   fn = f'./data/ilp_preds/{model}/{dataset}_k_{k_num}'
#   print(fn)
#   preds = assign_columns(model, q_scores, dq_scores, dataset, t_num, w, num_partitions, args.partition, fn)
#   # merge(num_partitions, fn, 'json')
#   # eval_preds(dataset, read_json(f'{fn}.json'))