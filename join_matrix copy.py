import os
import json
import pickle
from tqdm import tqdm
from groq import Groq
client = Groq(api_key="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")

def read_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)

def write_pickle(r, fn):
    with open(fn, 'wb') as f:
        pickle.dump(r, f)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_or_init_cache(path):
    if os.path.exists(path):
        return read_json(path)
    return {}

def check_join_exists(table1, table2, edges_data):
    print(f"{table1}-{table2}" in edges_data or f"{table2}-{table1}" in edges_data)
    return f"{table1}-{table2}" in edges_data or f"{table2}-{table1}" in edges_data

def get_table_schema(tables_data, table_key):
    if table_key in tables_data:
        table_info = tables_data[table_key]
        schema_info = {
            "table_name": table_info.get("table_name", ""),
            "columns": table_info.get("column_names_original", [])
        }
        return schema_info
    return None

def load_or_init_cache(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def get_join_relevance_llm(table1, table2, edges_data, query_text, dataset):
    import json
    
    tables = read_json(f'./jar2-main/data/{dataset}/dev_tables.json')
    
    join_key1 = f"{table1}-{table2}"
    join_key2 = f"{table2}-{table1}"
    
    if join_key1 in edges_data:
        join_key = join_key1
        possible_joins = edges_data[join_key1]
    elif join_key2 in edges_data:
        join_key = join_key2
        possible_joins = edges_data[join_key2]
    else:
        print(f"No join information found between {table1} and {table2}")
        return 0
    
    # Get schema details
    schema1 = get_table_schema(tables, table1)
    schema2 = get_table_schema(tables, table2)
    
    schema1_str = f"Table: {schema1['table_name']}\nColumns: {', '.join(schema1['columns'])}" if schema1 else f"No schema info for {table1}"
    schema2_str = f"Table: {schema2['table_name']}\nColumns: {', '.join(schema2['columns'])}" if schema2 else f"No schema info for {table2}"
    
    # One-shot example
    one_shot_example = """
### Example (One-shot)

Query: "what is the expense for category food for the september meeting event?"

Table Schema for expense:  
Table: expense  
Columns: expense_id, expense_date, cost, link_to_budget  

Table Schema for budget:  
Table: budget  
Columns: budget_id, category, spent, remaining, amount, event_status, event_name  

Possible joins between tables student_club#sep#budget and student_club#sep#expense are represented as:  
"db_id#sep#table_name#sep#column_name - db_id#sep#table_name#sep#column_name"  

{
  "student_club#sep#budget#sep#budget_id - student_club#sep#expense#sep#link_to_budget": {
    "join_category": "inferred PK-FK",
    "combined_score": 0.6723837485680213
  },
  "student_club#sep#budget#sep#spent - student_club#sep#expense#sep#cost": {
    "join_category": "Semantic",
    "combined_score": 0.5188322831844462
  }
}

Reasoning:  
- Query needs category + event_name (from budget) and cost (from expense).  
- To connect these, the join **budget_id = link_to_budget** is valid.  
Answer: **1**
"""   
    prompt = f"""
You are given a natural language query, schemas of two tables, and possible join conditions between them.  

Your task: Determine if **any** of the given join conditions is valid for answering the query.  
- If at least one join is valid → Respond with **"1"**.  
- If none of the joins are valid, or the tables are not relevant at all → Respond with **"0"**.  

Think step by step:  
1. Understand the query and what information it is asking for.  
2. Look at the schemas of both tables.  
3. Check the given possible joins.  
4. Decide if the query can actually be answered by using any of those joins.  

Do not assume anything outside the given information.  
Final output should be ONLY **1** or **0**, nothing else.  

{one_shot_example}

---

### Now, solve for the given input:

Query: "{query_text}"  

Table Schema for {table1.split('#sep#')[-1]}:  
{schema1_str}  

Table Schema for {table2.split('#sep#')[-1]}:  
{schema2_str}  

Possible joins between tables {table1} and {table2} are represented as:  
"db_id#sep#table_name#sep#column_name - db_id#sep#table_name#sep#column_name"  

{json.dumps(possible_joins, indent=2)}  

Respond with ONLY **1** or **0**.
"""
    
    print(prompt)  # For debugging / sending to LLM
    return prompt

# ------------------ Join Predictions ------------------
def get_join_relevance_llm(table_key1, table_key2, edges_data, query_text, dataset):


    # Load table schemas
    tables = read_json(f'./jar2-main/data/{dataset}/dev_tables.json')

    # Identify correct join key direction
    join_key1 = f"{table_key1}-{table_key2}"
    join_key2 = f"{table_key2}-{table_key1}"

    if join_key1 in edges_data:
        possible_joins = edges_data[join_key1]
    elif join_key2 in edges_data:
        possible_joins = edges_data[join_key2]
    else:
        return "NO"

    # Retrieve schema details
    schema1 = get_table_schema(tables, table_key1)
    schema2 = get_table_schema(tables, table_key2)

    schema1_str = f"Table: {schema1['table_name']}\nColumns: {', '.join(schema1['columns'])}" if schema1 else f"No schema info for {table_key1}"
    schema2_str = f"Table: {schema2['table_name']}\nColumns: {', '.join(schema2['columns'])}" if schema2 else f"No schema info for {table_key2}"

    # Build LLM prompt
    prompt = f"""
You are an expert in analyzing and validating join relationships between two database tables for a given natural language query.

You will be provided with:
- A **query** in natural language,
- The **schemas** of two tables,
- A set of **join candidates**, where:
  - Each table pair key is formatted as: db_id#sep#table1-db_id#sep#table2
  - Each column pair key is formatted as: db_id#sep#table1#sep#column1-db_id#sep#table2#sep#column2

Your goal is to determine whether any of the given join candidates represent a **valid join** that can correctly answer the query.

Respond using exactly **two lines**:
- Line 1: Explanation
- Line 2: Output "YES" if at least one join candidate is valid, otherwise "NO"

=====================
Query: "{query_text}"

Table Schema for {table_key1.split('#sep#')[-1]}:
{schema1_str}

Table Schema for {table_key2.split('#sep#')[-1]}:
{schema2_str}

Join Candidates:
{json.dumps(possible_joins, indent=2)}
=====================
"""

    # Call LLM
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-32b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=2000
        )

        llm_response = completion.choices[0].message.content.strip()
        print(llm_response)
        # Extract decision (second line)
        lines = [line.strip() for line in llm_response.split("\n") if line.strip()]
        if len(lines) < 2:
            return "NO"

        decision = lines[1].upper()
        if decision not in {"YES", "NO"}:
            return "NO"

        return decision

    except Exception:
        return "NO"



def join_dict_per_query(table_keys, edges_data, dataset, fn):
    qs = read_json(f'./jar2-main/data/{dataset}/dev.json')
    qs = qs[:1]
    cache_file = f'{fn}.json'
    join_cache = load_or_init_cache(cache_file)
    
    preds = []
    for q_idx, q in enumerate(tqdm(qs)):
        query_text = q["question"]
        print(f"\n=== Processing query {q_idx+1}/{len(qs)}: {query_text} ===")
        preds_for_Q = {}
        
        for table_key1 in table_keys:
            for table_key2 in table_keys:
                if table_key1 == table_key2:
                    continue
                if check_join_exists(table_key1, table_key2, edges_data):
                    join_key = f"{table_key1}-{table_key2}"
                    if join_key not in join_cache:
                        print(f"Initializing cache for {join_key}")
                        join_cache[join_key] = {}
                    if str(q_idx) not in join_cache[join_key]:
                        print(f"Query {q_idx} not in cache for {join_key}, calling LLM...")
                        join_cache[join_key][str(q_idx)] = check_join_exists(table_key1, table_key2, edges_data)
                        # get_join_relevance_llm(
                        #     table_key1, table_key2, edges_data, query_text, dataset
                        # )
                        save_json(cache_file, join_cache)
                    else:
                        print(f"Using cached result for query {q_idx}, join {join_key}")
                    preds_for_Q[join_key] = join_cache[join_key][str(q_idx)]
        
        preds.append(preds_for_Q)
    
    return preds


def main():
    dataset = "bird"  
    fn = "join_predictions_bird_edit_check"

    edges_data = read_json(f'./jar2-main/data/{dataset}/join_edges2.json')
    table_keys = list(read_json(f'./jar2-main/data/{dataset}/dev_tables.json').keys())
    
    qs = read_json(f'./jar2-main/data/{dataset}/dev.json') # only first 5 queries
    print(f"Starting join prediction for {len(table_keys)} tables and dataset '{dataset}' (first 5 queries)...")
    
    # Use a modified version of join_dict_per_query to accept a subset of queries
    cache_file = f'{fn}.json'
    join_cache = load_or_init_cache(cache_file)
    
    preds = []
    for q_idx, q in enumerate(tqdm(qs)):
        query_text = q["question"] 
        preds_for_Q = {}
        
        for table_key1 in table_keys:
            for table_key2 in table_keys:
                if table_key1 == table_key2:
                    continue
                
                if check_join_exists(table_key1, table_key2, edges_data):
                    join_key = f"{table_key1}-{table_key2}"
                    
                    if join_key not in join_cache:
                        join_cache[join_key] = {}
                    
                    if str(q_idx) not in join_cache[join_key]:
                        join_cache[join_key][str(q_idx)] = check_join_exists(table_key1, table_key2, edges_data)
                        
                        # get_join_relevance_llm(
                        #     table_key1, table_key2, edges_data, query_text, dataset
                        # )
                        save_json(cache_file, join_cache)
                    
                    preds_for_Q[join_key] = join_cache[join_key][str(q_idx)]
        
        preds.append(preds_for_Q)
    
    print("\nPredictions for first 5 queries completed.")
    print(json.dumps(preds, indent=2))

if __name__ == "__main__":
    main()


# def join_dict_all_queries(dataset, fn, edges_data, tables):
#     qs = read_json(f'./jar2-main/data/{dataset}/dev.json')
#     cache_file = f'{fn}.json'
#     join_cache = load_or_init_cache(cache_file)
    
#     preds = []
#     for q_idx, q in enumerate(tqdm(qs)):
#         query_text = q["question"] if isinstance(q, dict) else str(q)
        
#         # Print statement for every query
#         print(f"Processing query {q_idx + 1}/{len(qs)}: {query_text}")
        
#         preds_for_q = {}
        
#         for table_key1 in tables:
#             for table_key2 in tables:
#                 if table_key1 == table_key2:
#                     continue
#                 if check_join_exists(table_key1, table_key2, edges_data):
#                     join_key = f"{table_key1}-{table_key2}"
                    
#                     if join_key not in join_cache:
#                         join_cache[join_key] = {}
                    
#                     if str(q_idx) not in join_cache[join_key]:
#                         join_cache[join_key][str(q_idx)] = get_join_relevance_llm(
#                             table_key1, table_key2, edges_data, query_text, dataset
#                         )
#                         save_json(cache_file, join_cache)
                    
#                     preds_for_q[join_key] = join_cache[join_key][str(q_idx)]
        
#         preds.append(preds_for_q)
    
#     save_json(cache_file, join_cache)
#     return preds


# def get_join_relevance_llm(table1, table2, edges_data, query_text, dataset):
#     tables = read_json(f'./jar2-main/data/{dataset}/dev_tables.json')   
#     join_key1 = f"{table1}#sep#{table2}"
#     join_key2 = f"{table2}#sep#{table1}"   
#     if join_key1 in edges_data:
#         join_key = join_key1
#         possible_joins = edges_data[join_key1]
#     elif join_key2 in edges_data:
#         join_key = join_key2
#         possible_joins = edges_data[join_key2]
#     else:
#         return 0   
#     print(f"LLM call for join between {table1} and {table2} for query: {query_text}")  
#     prompt = f"""Query: "{query_text}"
# Table Schema for {table1.split('#sep#')[-1]}:
# """
#     schema1 = get_table_schema(tables, table1)
#     if schema1:
#         prompt += f"Table: {schema1['table_name']}\n"
#         prompt += f"Columns: {', '.join(schema1['columns'])}\n\n"    
#     prompt += f"Table Schema for {table2.split('#sep#')[-1]}:\n"   
#     schema2 = get_table_schema(tables, table2)
#     if schema2:
#         prompt += f"Table: {schema2['table_name']}\n"
#         prompt += f"Columns: {', '.join(schema2['columns'])}\n\n"
    
#     prompt += f"Possible joins between tables {table1} and {table2}:\n"
#     prompt += json.dumps(possible_joins, indent=2)
    
#     prompt += f"""

# The information shared above shows the joins possible between the tables {table1} and {table2}. 
# You have to tell based on the query, if any of the joins given above is a valid join for the query or not.

# Consider:
# 1. Whether any of the provided join columns is actually the correct join to answer the query correctly
# 2. Other calculated information has also been shared just for reference

# You have to respond with ONLY '1' if one of the joins provided above is a valid join for this query, 
# else return ONLY '0' if none of the join columns between the 2 tables given above is a valid join for the given query.

# Respond with only a single digit: 1 or 0
# """
#     print(prompt)
    
#     try:
#         response = client.chat.completions.create(
#             model="llama-3.1-70b-versatile",  # or "mixtral-8x7b-32768"
#             messages=[
#                 {
#                     "role": "system",
#                     "content": "You are a database expert. Respond with only '1' or '0'."
#                 },
#                 {
#                     "role": "user",
#                     "content": prompt
#                 }
#             ],
#             temperature=0.1,
#             max_tokens=10
#         )
        
#         llm_response = response.choices[0].message.content.strip()
#         print(f"LLM response for query '{query_text}' between {table1} and {table2}: {llm_response}")
        
#         if '1' in llm_response:
#             return 1
#         elif '0' in llm_response:
#             return 0
#         else:
#             print(f"Unexpected LLM response: {llm_response}, defaulting to 0")
#             return 0
            
#     except Exception as e:
#         print(f"LLM call failed: {e}")
#         return 0

# def join_dict_per_query(table_keys, edges_data, dataset, fn):
#     qs = read_json(f'./jar2-main/data/{dataset}/dev.json')
#     cache_file = f'{fn}.json'
#     join_cache = load_or_init_cache(cache_file)
    
#     preds = []
#     for q_idx, q in enumerate(tqdm(qs)):
#         query_text = q["question"] 
#         preds_for_Q = {}
        
#         for table_key1 in table_keys:
#             for table_key2 in table_keys:
#                 if table_key1 == table_key2:
#                     continue
                
#                 if check_join_exists(table_key1, table_key2, edges_data):
#                     join_key = f"{table_key1}-{table_key2}"
                    
#                     if join_key not in join_cache:
#                         join_cache[join_key] = {}
                    
#                     if str(q_idx) not in join_cache[join_key]:
#                         join_cache[join_key][str(q_idx)] = get_join_relevance_llm(
#                             table_key1, table_key2, edges_data, query_text, dataset
#                         )
#                         save_json(cache_file, join_cache)
                    
#                     preds_for_Q[join_key] = join_cache[join_key][str(q_idx)]
        
#         preds.append(preds_for_Q)
    
#     return preds





# # ------------------------
# # Groq LLM Join Relevance
# # ------------------------
# from groq import Groq

# def evaluate_join(table1, table2, query_text, client: Groq):
#     # Construct the prompt for the LLM
#     prompt = f"""
#     The information shared above shows the joins possible between the tables {table1} and {table2}.
#     You have to tell based on the query, if any of the joins given above is a valid join for the query or not.

#     Consider:
#     1. Whether any of the provided join columns is actually the correct join to answer the query correctly
#     2. Other calculated information has also been shared just for reference

#     Respond with ONLY '1' if one of the joins provided above is a valid join for this query,
#     else return ONLY '0' if none of the join columns between the 2 tables given above is a valid join for the given query.

#     Respond with only a single digit: 1 or 0
#     Query: {query_text}
#     """

#     completion = client.chat.completions.create(
#         model="openai/gpt-oss-20b",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=1,
#         max_completion_tokens=8192,
#         top_p=1,
#         reasoning_effort="medium",
#         stream=True
#     )


#     response_text = ""
#     for chunk in completion:
#         response_text += chunk.choices[0].delta.content or ""


#     print(f"LLM response for tables ({table1}, {table2}) and query '{query_text}':")
#     print(response_text.strip())

#     try:
#         prediction = int(response_text.strip())
#     except ValueError:
#         print("Warning: LLM response could not be converted to int, defaulting to 0")
#         prediction = 0

#     return prediction

