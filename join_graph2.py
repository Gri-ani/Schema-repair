
import networkx as nx
import json
import matplotlib as plt

def read_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)
    

class JoinabilityGraphs:
    def __init__(self, dataset):
        self.dataset = dataset
        self.tables = read_json(f'./jar2-main/data/{dataset}/dev_tables.json')
        self.jaccard = read_json(f'./jar2-main/data/{dataset}/dev_jaccard.json')
        self.semantic_col_sim = read_json(f'./jar2-main/data/{dataset}/semantic_col_sim(1).json')
        self.exact_col_sim = read_json(f'./jar2-main/data/{dataset}/exact_col_sim(2).json')
        self.uniqueness = read_json(f'./jar2-main/data/{dataset}/dev_uniqueness.json')

        self.graphs = {}
        self.build_graphs()

    def is_compatible_type(self, type1, type2):
        type1 = type1.lower()
        type2 = type2.lower()

        if type1 == type2:
            return True
        
        numeric_types = {'integer', 'real', 'number', 'numeric', 'float', 'double', 'int'}
        if type1 in numeric_types and type2 in numeric_types:
            return True
        
        text_types = {'text', 'varchar', 'char', 'string'}
        if type1 in text_types and type2 in text_types:
            return True
        
        datetime_types = {'date', 'time', 'datetime', 'timestamp'}
        if type1 in datetime_types and type2 in datetime_types:
            return True
        
        return False
    
    def build_graphs(self):
        print("building graphs.")
        surrogate_keys = {'id', 'row_id', 'pk', 'key', 'index', 'auto_id'}
        # jaccard_min_threshold = 0.05

        for table_key, table_meta in self.tables.items():
            db_id = table_meta["db_id"]
            if db_id not in self.graphs:
                self.graphs[db_id] = nx.MultiGraph()
                print(f"created graph for database: {db_id}")

            self.graphs[db_id].add_node(
                table_key,
                table_name=table_meta['table_name_original']
            )
            print(f"Added node: {table_key} to DB graph {db_id}")

        seen_pairs = set()
        for table_key, table_meta in self.tables.items():
            db_id = table_meta["db_id"]
            cols1 = table_meta["column_names_original"]
            types1 = table_meta["column_types"]

            for table_key2, table_meta2 in self.tables.items():
                if table_key == table_key2 or table_meta["db_id"] != table_meta2["db_id"]:
                    continue

                pair = tuple(sorted([table_key, table_key2]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                cols2 = table_meta2["column_names_original"]
                types2 = table_meta2["column_types"]

                print(f"\nComparing table '{table_key}' with table '{table_key2}' in DB {db_id}")
                print(f"Columns in {table_key}: {cols1}")
                print(f"Columns in {table_key2}: {cols2}")

                edges = []
                table_pair_key = f'{table_key}-{table_key2}'

                for i, c1 in enumerate(cols1):
                    if c1.lower() in surrogate_keys:
                        print(f"  Skipping col {c1} in {table_key} (surrogate key)")
                        continue

                    for j, c2 in enumerate(cols2):
                        if c2.lower() in surrogate_keys:
                            print(f"  Skipping col {c2} in {table_key2} (surrogate key)")
                            continue

                  
                        if not self.is_compatible_type(types1[i], types2[j]):
                            print(f"  Skipping pair ({c1}, {c2}) - incompatible types {types1[i]} vs {types2[j]}")
                            continue

                        cols_key1 = f'{table_key}#sep#{c1}'
                        cols_key2 = f'{table_key2}#sep#{c2}'

                        pair_key1 = f'{cols_key1}-{cols_key2}'
                        pair_key2 = f'{cols_key2}-{cols_key1}'

                        # Similarity scores
                        jaccard_score = 0.0
                        semantic_score = 0.0
                        exact_score = 0.0

                        if table_pair_key in self.jaccard:
                            jaccard_score = self.jaccard[table_pair_key].get(pair_key1, 0.0)
                            if jaccard_score == 0.0:
                                jaccard_score = self.jaccard[table_pair_key].get(pair_key2, 0.0)
                    
                        is_numeric = types1[i].lower() in {'integer', 'real', 'number', 'numeric', 'float', 'double', 'int'}
                        # if not is_numeric and jaccard_score < jaccard_min_threshold:
                        #     print(f"  Skipping pair ({c1}, {c2}) - low Jaccard {jaccard_score}")
                        #     continue
                    
                        if table_pair_key in self.semantic_col_sim:
                            semantic_score = self.semantic_col_sim[table_pair_key].get(pair_key1, 0.0)
                            if semantic_score == 0.0:
                                semantic_score = self.semantic_col_sim[table_pair_key].get(pair_key2, 0.0)

                        if table_pair_key in self.exact_col_sim:
                            exact_score = self.exact_col_sim[table_pair_key].get(pair_key1, 0.0)
                            if exact_score == 0.0:
                                exact_score = self.exact_col_sim[table_pair_key].get(pair_key2, 0.0)
                    
                        uniqueness_col1 = self.uniqueness.get(f'{table_key}#sep#{c1}', 0)
                        uniqueness_col2 = self.uniqueness.get(f'{table_key2}#sep#{c2}', 0)
                        uniqueness = max(uniqueness_col1, uniqueness_col2)

                        combined_score = 0.4 * jaccard_score + 0.3 * semantic_score + 0.3 * exact_score
                        min_threshold = 0.4

                        if combined_score < min_threshold:
                            print(f"  Skipping pair ({c1}, {c2}) - combined score too low {combined_score}")
                            continue

                        edge_info = {
                            "col1": c1,
                            "col2": c2,
                            "jaccard": jaccard_score,
                            "uniqueness": uniqueness,
                            "combined_score": combined_score
                        }
                        edges.append(edge_info)
                        print(f"  Adding edge: {edge_info}")

                for edge_info in edges:
                    self.graphs[db_id].add_edge(table_key, table_key2, **edge_info)   

                if edges:
                    print(f"Added {len(edges)} edges between {table_key} and {table_key2} in DB {db_id}")  
                else:
                    print(f"No edges added between {table_key} and {table_key2}") 

    # def classify_join(self, uniqueness, jaccard_score, semantic_score, exact_score):
    #     high_jaccard = 0.7
    #     high_semantic = 0.7
    #     high_exact = 0.8

    #     # if (types1.lower() in {'integer', 'real', 'number', 'numeric', 'float', 'double', 'int'} and  
    #     #     types2.lower() in {'integer', 'real', 'number', 'numeric', 'float', 'double', 'int'}) and \
    #     #     semantic_score > high_semantic and jaccard_score < 0.2:
    #     #     return "numeric type with high semantic match"

    #     if uniqueness >= 0.95 and (semantic_score > 0.8 or jaccard_score > 0.8):
    #         return "inferred PK-FK"
        
    #     elif jaccard_score >= high_jaccard:
    #         return "Value-Based"
        
    #     elif exact_score >= high_exact:
    #         return "Exact match"
        
    #     elif semantic_score >= high_semantic:
    #         return "Semantic"
        
    #     else:
    #         return "Weak"

    def get_graph(self, db_id):
        return self.graphs.get(db_id)
    
    def print_graph_summary(self, db_id=None):
        graphs_to_check = {db_id: self.graphs[db_id]} if db_id else self.graphs
        
        for db_name, graph in graphs_to_check.items():
            print(f"\nGraph Summary for {db_name}")
            print(f"Number of tables (nodes): {graph.number_of_nodes()}")
            print(f"Number of join paths (edges): {graph.number_of_edges()}")
            
            category_counts = {}
            for _, _, data in graph.edges(data=True):
                category = data.get('join_category', 'Unknown')
                category_counts[category] = category_counts.get(category, 0) + 1
            
            print("\nEdges by join category:")
            for category, count in sorted(category_counts.items()):
                print(f"  {category}: {count}")

    def has_path(self, db_id, table1, table2):
        if db_id not in self.graphs:
            return False
        
        graph = self.graphs[db_id]
        if table1 not in graph.nodes() or table2 not in graph.nodes():
            return False
        
        return nx.has_path(graph, table1, table2)
    
    def get_shortest_path(self, db_id, table1, table2):
        if db_id not in self.graphs:
            print(f"Database {db_id} not found")
            return None
        
        graph = self.graphs[db_id]
        if table1 not in graph.nodes():
            print(f"Table {table1} not found in database {db_id}")
            return None
        if table2 not in graph.nodes():
            print(f"Table {table2} not found in database {db_id}")
            return None
        
        try:
            return nx.shortest_path(graph, table1, table2)
        except nx.NetworkXNoPath:
            print(f"No path exists between {table1} and {table2}")
            return None
        
    def get_all_paths(self, db_id, table1, table2, cutoff=None):
        if db_id not in self.graphs:
            return []
        
        graph = self.graphs[db_id]
        if table1 not in graph.nodes() or table2 not in graph.nodes():
            return []
        
        try:
            return list(nx.all_simple_paths(graph, table1, table2, cutoff=cutoff))
        except nx.NetworkXNoPath:
            return []
        
    def is_connected(self, db_id):
        if db_id not in self.graphs:
            return False
        return nx.is_connected(self.graphs[db_id])
    
    def get_connected_components(self, db_id):
        if db_id not in self.graphs:
            return []
        return list(nx.connected_components(self.graphs[db_id]))

    def save_edges_to_json(self, filepath):
        
        grouped_edges = {}

        for db_id, graph in self.graphs.items():
            for u, v, data in graph.edges(data=True):
                table_pair_key = f"{u}-{v}"
                col_pair_key = f"{u}#sep#{data.get('col1')}-{v}#sep#{data.get('col2')}"

                if table_pair_key not in grouped_edges:
                    grouped_edges[table_pair_key] = {}

                grouped_edges[table_pair_key][col_pair_key] = {
                    "jaccard": data.get("jaccard"),
                    # "semantic": data.get("semantic"),
                    # "exact_col_match": data.get("exact_col_match"),
                    "uniqueness": data.get("uniqueness"),
                    # "join_category": data.get("join_category"),
                    "combined_score": data.get("combined_score")
                    # "type1": data.get("type1"),
                    # "type2": data.get("type2")
                }

        with open(filepath, "w") as f:
            json.dump(grouped_edges, f, indent=2)

        print(f" Saved grouped join edges to {filepath}")

   
    # def save_edges_to_json(self, filepath):
        
    #     all_edges = []

    #     for db_id, graph in self.graphs.items():
    #         for u, v, data in graph.edges(data=True):
    #             edge_record = {
    #                 "db_id": db_id,
    #                 "table1": u,
    #                 "table2": v,
    #                 "col1": data.get("col1"),
    #                 "col2": data.get("col2"),
    #                 "jaccard": data.get("jaccard"),
    #                 "semantic": data.get("semantic"),
    #                 "exact_col_match": data.get("exact_col_match"),
    #                 "uniqueness": data.get("uniqueness"),
    #                 "join_category": data.get("join_category"),
    #                 "combined_score": data.get("combined_score"),
    #                 "type1": data.get("type1"),
    #                 "type2": data.get("type2")
    #             }
    #             all_edges.append(edge_record)

    #     with open(filepath, "w") as f:
    #         json.dump(all_edges, f, indent=2)
        
    #     print(f"Saved {len(all_edges)} join edges to {filepath}")

def main():
    dataset = "bird"  
    
    print("="*80)
    print(f"Building Joinability Graphs for dataset: {dataset}")
    print("="*80)

    try:
        jg = JoinabilityGraphs(dataset=dataset)
        print("\nGraph building completed successfully!")
    except Exception as e:
        print(f"\nError building graphs: {e}")
        return

    print("\n" + "="*80)
    print("GRAPH SUMMARIES")
    print("="*80)
    jg.print_graph_summary()
    jg.save_edges_to_json(f"./jar2-main/data/{dataset}/join_edges3.json")
    db_id = "student_club"
    print(jg.has_path( db_id, 'student_club#sep#member', 'student_club#sep#college'))
    


if __name__ == "__main__":
    main()
