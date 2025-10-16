
import sqlite3
import json
from tqdm import tqdm
from collections import defaultdict
from utils import read_json, write_json, is_compatible_type


class JaccardCalculator:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.tables = self.read_json(f'./jar2-main/data/{dataset}/dev_tables.json')
        self.output_fn = f'./jar2-main/data/{dataset}/dev_jaccard_inverted.json'
        self.jaccard = {}

    def compute_jaccard(self):
        print("Building inverted index...")
        inverted_index = defaultdict(set)
        column_values = {}

        for t_key, t_info in tqdm(self.tables.items(), desc="Extracting columns"):
            db = t_info['db_id']
            conn = sqlite3.connect(f'./jar2-main/data/{self.dataset}/dev_databases/{db}/{db}.sqlite')
            cur = conn.cursor()
            t_name = t_info['table_name_original']
            cols = t_info['column_names_original']
            types = t_info['column_types']

            for col, ctype in zip(cols, types):
                try:
                    cur.execute(f"SELECT DISTINCT `{col}` FROM `{t_name}` WHERE `{col}` IS NOT NULL;")
                    vals = set(str(v[0]).strip().lower() for v in cur.fetchall() if v[0] is not None)
                except Exception as e:
                    print(f"Error reading {t_name}.{col}: {e}")
                    continue

                col_key = f"{t_key}#sep#{col}"
                column_values[col_key] = {"values": vals, "type": ctype}

                for v in vals:
                    inverted_index[v].add(col_key)

            conn.close()

        # Inverted index
        print("intersections usingf inverted indexes")
        intersection_counts = defaultdict(int)
        for val, col_keys in tqdm(inverted_index.items(), desc="Index scanning"):
            col_keys = list(col_keys)
            for i in range(len(col_keys)):
                for j in range(i + 1, len(col_keys)):
                    a, b = col_keys[i], col_keys[j]
                    print(a)
                    db1, t1, c1 = a.split("#sep#")
                    db2, t2, c2 = b.split("#sep#")
                    t1 = f'{db1}#sep#{t1}'
                    t2 = f'{db2}#sep#{t2}'
                    if self.tables[t1]["db_id"] != self.tables[t2]["db_id"]:
                        continue  
                    if not self.is_compatible_type(column_values[a]["type"], column_values[b]["type"]):
                        continue
                    intersection_counts[(a, b)] += 1

       
        # print("Compu Jaccard scores.")
        for (a, b), inter in tqdm(intersection_counts.items(), desc="Scoring"):
            db1, t1, c1 = a.split("#sep#")
            db2, t2, c2 = b.split("#sep#")

            table_pair_key1 = f"{db1}#sep#{t1}-{db2}#sep#{t2}"
            table_pair_key2 = f"{db2}#sep#{t2}-{db1}#sep#{t1}"
           
            if table_pair_key1 in self.jaccard:
               
                table_pair_key = table_pair_key1
            elif(table_pair_key2 in self.jaccard):
                table_pair_key = table_pair_key2

            else:
                table_pair_key = table_pair_key1
                self.jaccard[table_pair_key] = {}


            n1 = len(column_values[a]["values"])
            n2 = len(column_values[b]["values"])
            denom = n1 + n2 - inter
            score = inter / denom if denom > 0 else 0.0

            # if table_pair_key not in self.jaccard:
                

            col_pair_key = f"{a}-{b}"
            self.jaccard[table_pair_key][col_pair_key] = score


        self.write_json(self.jaccard, self.output_fn)
        print(f"Jaccard computation (Inverted Index) - {self.output_fn}")
        return self.jaccard

if __name__ == "__main__":
  
    dataset_name = "bird"
    calculator = JaccardCalculator(dataset=dataset_name)
    jaccard_results = calculator.compute_jaccard()
    
    # print("\nSample Jaccard results:")
    # for table_pair, cols in list(jaccard_results.items())[:5]:
    #     print(f"\nTable pair: {table_pair}")
    #     for col_pair, score in list(cols.items())[:5]:
    #         print(f"  {col_pair}: {score:.4f}")



# import sqlite3
# import json
# from tqdm import tqdm

# class JaccardCalculator:
#     def __init__(self, dataset: str):
#         self.dataset = dataset
#         self.tables = self.read_json(f'./jar2-main/data/{dataset}/dev_tables.json')
#         self.output_fn = f'./jar2-main/data/{dataset}/dev_jaccard2.json'
#         self.jaccard = {}

#     def read_json(fn):
#         with open(fn, 'r', encoding='utf-8') as f:
#             return json.load(f)

#     def write_json(obj, fn):
#         with open(fn, 'w', encoding='utf-8') as f:
#             json.dump(obj, f, indent=2)

#     def is_compatible_type(type1, type2):
#         type1 = type1.lower()
#         type2 = type2.lower()

#         if type1 == type2:
#             return True
        
#         numeric_types = {'integer', 'real', 'number', 'numeric', 'float', 'double', 'int'}
#         if type1 in numeric_types and type2 in numeric_types:
#             return True
        
#         text_types = {'text', 'varchar', 'char', 'string'}
#         if type1 in text_types and type2 in text_types:
#             return True
        
#         datetime_types = {'date', 'time', 'datetime', 'timestamp'}
#         if type1 in datetime_types and type2 in datetime_types:
#             return True
        
#         return False

#     def compute_jaccard(self):
#         for t1 in tqdm(self.tables, desc="Tables"):
#             for t2 in self.tables:
#                 if t1 == t2:
#                     continue

#                 db1, db2 = self.tables[t1]['db_id'], self.tables[t2]['db_id']
#                 if db1 != db2:
#                     continue  

#                 if f'{t1}-{t2}' in self.jaccard or f'{t2}-{t1}' in self.jaccard:
#                     continue 

#                 table_pair_key = f'{t1}-{t2}'
#                 self.jaccard[table_pair_key] = {}

#                 conn = sqlite3.connect(f'./jar2-main/data/{self.dataset}/dev_databases/{db1}/{db1}.sqlite')
#                 cur = conn.cursor()

#                 t_name1, t_name2 = self.tables[t1]['table_name_original'], self.tables[t2]['table_name_original']
#                 cols1, types1 = self.tables[t1]['column_names_original'], self.tables[t1]['column_types']
#                 cols2, types2 = self.tables[t2]['column_names_original'], self.tables[t2]['column_types']

#                 for c1, type1 in zip(cols1, types1):
#                     cur.execute(f'SELECT COUNT(DISTINCT `{c1}`) FROM `{t_name1}` WHERE `{c1}` IS NOT NULL;')
#                     n1 = cur.fetchone()[0]
#                     if n1 == 0:
#                         continue

#                     for c2, type2 in zip(cols2, types2):
#                         cur.execute(f'SELECT COUNT(DISTINCT `{c2}`) FROM `{t_name2}` WHERE `{c2}` IS NOT NULL;')
#                         n2 = cur.fetchone()[0]
#                         if n2 == 0:
#                             continue
#                         if not self.is_compatible_type(type1, type2):
#                             continue
#                         q = f"""
#                             SELECT COUNT(DISTINCT a.`{c1}`)
#                             FROM `{t_name1}` a
#                             JOIN `{t_name2}` b
#                               ON a.`{c1}` = b.`{c2}`
#                             WHERE a.`{c1}` IS NOT NULL AND b.`{c2}` IS NOT NULL;
#                         """
#                         cur.execute(q)
#                         inter = cur.fetchone()[0]

#                         denom = n1 + n2 - inter
#                         sim_score = inter / denom if denom > 0 else 0.0
#                         col_pair_key = f'{t1}#sep#{c1}-{t2}#sep#{c2}'
#                         self.jaccard[table_pair_key][col_pair_key] = sim_score

#                 conn.close()
#                 self.write_json(self.jaccard, self.output_fn)

#         print(f"Jaccard computation finished. Saved to {self.output_fn}")
#         return self.jaccard


# # if __name__ == "__main__":
# #     dataset = "bird"
# #     jc = JaccardCalculator(dataset)
# #     jc.compute_jaccard()
# # import sqlite3
# # import json
# # from tqdm import tqdm
# # from datasketch import MinHash, MinHashLSH

# # class JaccardMinHashCalculator:
# #     def __init__(self, dataset: str, num_perm: int = 128, lsh_threshold: float = 0.3):
# #         self.dataset = dataset
# #         self.tables = self.read_json(f'./jar2-main/data/{dataset}/dev_tables.json')
# #         self.output_fn = f'./jar2-main/data/{dataset}/dev_jaccard_minhash.json'
# #         self.num_perm = num_perm
# #         self.lsh_threshold = lsh_threshold
# #         self.jaccard = {}

# #     @staticmethod
# #     def read_json(fn):
# #         with open(fn, 'r', encoding='utf-8') as f:
# #             return json.load(f)

# #     @staticmethod
# #     def write_json(obj, fn):
# #         with open(fn, 'w', encoding='utf-8') as f:
# #             json.dump(obj, f, indent=2)

# #     @staticmethod
# #     def is_compatible_type(type1, type2):
# #         type1 = type1.lower()
# #         type2 = type2.lower()
# #         if type1 == type2:
# #             return True
# #         numeric_types = {'integer', 'real', 'number', 'numeric', 'float', 'double', 'int'}
# #         text_types = {'text', 'varchar', 'char', 'string'}
# #         datetime_types = {'date', 'time', 'datetime', 'timestamp'}
# #         if type1 in numeric_types and type2 in numeric_types:
# #             return True
# #         if type1 in text_types and type2 in text_types:
# #             return True
# #         if type1 in datetime_types and type2 in datetime_types:
# #             return True
# #         return False

# #     def build_minhash(self, conn, table, column, sample_rate=0.05):
# #         """Build MinHash sketch for a single column (sampled)."""
# #         m = MinHash(num_perm=self.num_perm)
# #         cur = conn.cursor()
# #         # Simple random sampling: hash-based
# #         cur.execute(f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL;")
# #         for i, (val,) in enumerate(cur):
# #             if val is None:
# #                 continue
# #             if hash(str(val)) % int(1/sample_rate) != 0:
# #                 continue
# #             m.update(str(val).strip().encode('utf-8'))
# #         return m

# #     def compute(self, sample_rate=0.05, verify_exact=True):
# #         """Compute candidate column pairs using MinHash, optionally verify exact Jaccard."""
# #         conn_cache = {}
# #         sketches = {}  # minhash sketch per column
# #         conn_objs = {}

# #         # Build sketches for all columns
# #         for t_name, table_info in tqdm(self.tables.items(), desc="Building MinHash"):
# #             db_id = table_info['db_id']
# #             conn_key = f"{db_id}"
# #             if conn_key not in conn_cache:
# #                 conn_cache[conn_key] = sqlite3.connect(f'./jar2-main/data/{self.dataset}/dev_databases/{db_id}/{db_id}.sqlite')
# #             conn = conn_cache[conn_key]

# #             cols, types = table_info['column_names_original'], table_info['column_types']
# #             for c, t in zip(cols, types):
# #                 sketches[f'{t_name}.{c}'] = (self.build_minhash(conn, table_info['table_name_original'], c, sample_rate), t)
# #             conn_objs[conn_key] = conn

# #         # Use LSH to generate candidate pairs
# #         lsh = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm)
# #         for col_key, (m, _) in sketches.items():
# #             lsh.insert(col_key, m)

# #         candidate_pairs = set()
# #         for col_key, (m, type1) in sketches.items():
# #             for match in lsh.query(m):
# #                 if match != col_key:
# #                     # type filter
# #                     _, type2 = sketches[match]
# #                     if not self.is_compatible_type(type1, type2):
# #                         continue
# #                     candidate_pairs.add(tuple(sorted([col_key, match])))

# #         print(f"Found {len(candidate_pairs)} candidate pairs via MinHash + LSH.")

# #         # Optional: exact verification
# #         if verify_exact:
# #             for a, b in tqdm(candidate_pairs, desc="Exact verification"):
# #                 t1, c1 = a.split('.', 1)
# #                 t2, c2 = b.split('.', 1)
# #                 db1 = self.tables[t1]['db_id']
# #                 conn = conn_objs[db1]
# #                 t_name1 = self.tables[t1]['table_name_original']
# #                 t_name2 = self.tables[t2]['table_name_original']

# #                 # Get distinct counts
# #                 cur = conn.cursor()
# #                 cur.execute(f'SELECT COUNT(DISTINCT `{c1}`) FROM `{t_name1}` WHERE `{c1}` IS NOT NULL;')
# #                 n1 = cur.fetchone()[0]
# #                 cur.execute(f'SELECT COUNT(DISTINCT `{c2}`) FROM `{t_name2}` WHERE `{c2}` IS NOT NULL;')
# #                 n2 = cur.fetchone()[0]

# #                 if n1 == 0 or n2 == 0:
# #                     continue

# #                 # Exact intersection
# #                 q = f"""
# #                     SELECT COUNT(DISTINCT a.`{c1}`)
# #                     FROM `{t_name1}` a
# #                     JOIN `{t_name2}` b
# #                       ON a.`{c1}` = b.`{c2}`
# #                     WHERE a.`{c1}` IS NOT NULL AND b.`{c2}` IS NOT NULL;
# #                 """
# #                 cur.execute(q)
# #                 inter = cur.fetchone()[0]
# #                 denom = n1 + n2 - inter
# #                 sim_score = inter / denom if denom > 0 else 0.0

# #                 table_pair_key = f'{t1}-{t2}'
# #                 col_pair_key = f'{t1}#sep#{c1}-{t2}#sep#{c2}'
# #                 if table_pair_key not in self.jaccard:
# #                     self.jaccard[table_pair_key] = {}
# #                 self.jaccard[table_pair_key][col_pair_key] = sim_score

# #             # Save results
# #             self.write_json(self.jaccard, self.output_fn)
# #             print(f"Jaccard with MinHash candidates saved to {self.output_fn}")

# #         # Close connections
# #         for conn in conn_objs.values():
# #             conn.close()

# #         return self.jaccard

# # if __name__ == "__main__":
# #     dataset = "bird"
# #     jc = JaccardMinHashCalculator(dataset, num_perm=128, lsh_threshold=0.3)
# #     jc.compute(sample_rate=0.05, verify_exact=True)