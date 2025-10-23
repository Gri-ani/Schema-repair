# import networkx as nx
# from join_graph2 import JoinabilityGraphs

# dataset = "bird"
# jg = JoinabilityGraphs(dataset=dataset)
# print("\nGraph building completed successfully!")

# def analyze_query_paths(G, queries):
#     results = []

#     for q in queries:
#         paths = q.get("path",[])
#         if not paths or len(paths[0]) == 0:
#             continue

#         raw_tables = [p for p in paths[0]]
#         db_id = raw_tables[0].split("#sep#")[0]
#         unique_tables = list(set(raw_tables))

#         if len(unique_tables) == 1:
#             continue

#         elif len(unique_tables)==2:
#             t1, t2 = unique_tables
#             if jg.has_path(db_id, t1, t2):
#                 sp = jg.get_shortest_path(db_id, t1, t2)
#                 edges = [(sp[i], sp[i+1]) for i in range(len(sp)-1)]
#                 results.append({
#                     "query": q["query"],
#                     "edges_to_check": edges
#                 })

#             else:
#                 results.append({
#                     "query": q["query"],
#                     "edges_to_check": edges
#                 })


#     # def get_connected_components(self, db_id):
#     #     if db_id not in self.graphs:
#     #         return []
#     #     return list(nx.connected_components(self.graphs[db_id]))
#         else:
#             components = [c for c in jg.get_connected_components(db_id)]
#             table_components = [next((c for c in components if t in c)for t in unique_tables)]
#             # find all connected components
#             map = {}

#             for t, comp in zip(unique_tables, table_components):
#                 if comp not in map:
#                     map[comp] = []

#                 map[comp].append(t)
            
#             subpaths = []
#             all_edges_to_check = []

#             for comp, tables in map.items():
#                 if(len(tables) == 1):
#                     subpaths.append(tables[0])
#                     all_edges_to_check.append(tables[0])

#                 else:
#                     subgraph_edges = []
#                     for i in range(len(tables)):
#                         for j in range(i+1, len(tables)):
#                             if jg.has_path(db_id, tables[i], tables[j]):
#                                 path = jg.get_shortest_path(db_id, tables[i], tables[j])
#                                 edges = [(path[k], path[k+1]) for k in range(len(path)-1)]
#                                 subgraph_edges.extend(edges)
#                                 all_edges_to_check.extend(edges)

#                             else:
#                                 all_edges_to_check.append([tables[i], tables[j]])
                    
#                     subgraph_edges = list(set(subgraph_edges))
#                     subpaths.append(subgraph_edges)
#             results.append({
#                 "query":q["query"],
#                 "optimized_paths": subpaths,
#                 "edges_to_check": all_edges_to_check
#             })

        
#     return results




#                 # def has_path(self, db_id, table1, table2):

# import json
# import networkx as nx
# from join_graph2 import JoinabilityGraphs

# # Initialize the joinability graphs
# dataset = "bird"
# jg = JoinabilityGraphs(dataset=dataset)
# print("\nGraph building completed successfully!")


# def analyze_query_paths(G, queries):
#     results = []

#     for q in queries:
#         paths = q.get("path", [])
#         if not paths or len(paths[0]) == 0:
#             # results.append({
#             #     "query": q["query"],
#             #     "edges_to_check": []
#             # })
#             continue

#         # Extract table names
#         raw_tables = [p for p in paths[0]]
#         db_id = raw_tables[0].split("#sep#")[0]
#         unique_tables = list(set(raw_tables))

#         # Single table
#         if len(unique_tables) == 1:
#             continue

#         # Two tables
#         elif len(unique_tables) == 2:
#             t1, t2 = unique_tables
#             if jg.has_path(db_id, t1, t2):
#                 sp = jg.get_shortest_path(db_id, t1, t2)
#                 edges = [(sp[i], sp[i+1]) for i in range(len(sp)-1)]
#                 results.append({
#                     "query": q["query"],
#                     "edges_to_check": edges
#                 })
#             else:
#                 results.append({
#                     "query": q["query"],
#                     "edges_to_check": [[t1], [t2]]
#                 })


#         else:
#             components = jg.get_connected_components(db_id)
#             table_components = [next((c for c in components if t in c), None) for t in unique_tables]
            
#             # for t in unique_tables:
#             #     comp = next((c for c in components if t in c), None)
#             #     if comp is None:
#             #         table_components.append(frozenset([t]))
#             #     else:
#             #         table_components.append(frozenset(comp))
#             comp_map = {}
#             for t, cp in zip(unique_tables, table_components):
#                 comp = frozenset(cp)
#                 if comp not in comp_map:
#                     comp_map[comp] = []
#                 comp_map[comp].append(t)

#             subpaths = []
#             all_edges_to_check = []

#             for comp, tables in comp_map.items():
#                 if len(tables) == 1:
#                     subpaths.append([tables[0]])
#                     all_edges_to_check.append([tables[0]])
#                 else:
#                     subgraph_edges = []
#                     for i in range(len(tables)):
#                         for j in range(i + 1, len(tables)):
#                             if jg.has_path(db_id, tables[i], tables[j]):
#                                 path = jg.get_shortest_path(db_id, tables[i], tables[j])
#                                 edges = [(path[k], path[k+1]) for k in range(len(path)-1)]
#                                 subgraph_edges.extend(edges)
#                                 all_edges_to_check.extend(edges)
#                             else:
#                                 all_edges_to_check.append((tables[i], tables[j]))

#                     subgraph_edges = list(set(subgraph_edges))
#                     subpaths.append(subgraph_edges)

#             results.append({
#                 "query": q["query"],
#                 "optimized_paths": subpaths,
#                 "edges_to_check": all_edges_to_check
#             })

#     return results

# def main():

#     json_file = r"jar2-main\data\bird\possible_table_paths_only.json"


#     with open(json_file, "r", encoding="utf-8") as f:
#         queries = json.load(f)


#     results = analyze_query_paths(jg, queries)


#     for res in results:
#         print("\nQuery:", res["query"])
#         if "optimized_paths" in res:
#             print("Optimized Paths:", res["optimized_paths"])
#         print("Edges to Check:", res["edges_to_check"])
    
#     output_file = r"jar2-main\data\bird\analyzed_query_paths.json"
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=4)

#     print(f"\nAnalysis complete! Results saved to: {output_file}")

# if __name__ == "__main__":
#     main()
import json
from join_graph2 import JoinabilityGraphs

dataset = "bird"
jg = JoinabilityGraphs(dataset=dataset)
print("\nGraph building completed successfully!")



with open("./jar2-main/data/bird/join_edges3.json", "r") as f:
    join_data = json.load(f)


def get_possible_joins(t1, t2, dataset_prefix):

    key1 = f"{dataset_prefix}#sep#{t1}-{dataset_prefix}#sep#{t2}"
    key2 = f"{dataset_prefix}#sep#{t2}-{dataset_prefix}#sep#{t1}"

    possible_joins = []

    if key1 in join_data:
        mappings = join_data[key1]
    elif key2 in join_data:
        mappings = join_data[key2]
    else:
        return []

    for pair, scores in mappings.items():
        cols = pair.split("-")
        left_col = cols[0].split("#sep#")[-1]
        right_col = cols[1].split("#sep#")[-1]

        possible_joins.append({
            "table1_column": f"{t1}.{left_col}",
            "table2_column": f"{t2}.{right_col}",
            "jaccard": scores.get("jaccard"),
            "uniqueness": scores.get("uniqueness")
        })

    return possible_joins


def analyze_query_paths(G, queries, subquery_mappings):
    results = []

    for q in queries:
        query_text = q["query"]


        mapping_entry = next((m for m in subquery_mappings if m["query"] == query_text), None)
        subquery_tables = {}
        if mapping_entry:
            for sub in mapping_entry["sub_queries_mapping"]:
                table_columns_map = {}
                for tm in sub["top_matches"]:
                    table = tm["table"].split('#sep#')[-1]
                    col = tm["column"]
                    if table not in table_columns_map:
                        table_columns_map[table] = []
                    table_columns_map[table].append(col)
                subquery_tables[sub["sub_query"]] = table_columns_map

        paths = q.get("path", [])
        if not paths or len(paths[0]) == 0:
            results.append({
                "query": query_text,
                "db_id": None,
                "optimized_paths": [],
                "edges_to_check": [],
                "possible_joins_for_edges": {},
                "subquery_tables": subquery_tables
            })
            continue

        # raw_tables = [p for p in paths[0]]
        raw_tables = paths 

        db_id = raw_tables[0].split("#sep#")[0]
        unique_tables = list(set(raw_tables))
        table_names = [t.split("#sep#")[-1] for t in raw_tables]

        if len(unique_tables) == 1:
            results.append({
                "query": query_text,
                "db_id": db_id,
                "optimized_paths": table_names,
                "edges_to_check": [table_names],
                "possible_joins_for_edges": {},
                "subquery_tables": subquery_tables
            })
            continue

        elif len(unique_tables) == 2:
            t1, t2 = [t.split("#sep#")[-1] for t in unique_tables]
            if G.has_path(db_id, unique_tables[0], unique_tables[1]):
                sp = G.get_shortest_path(db_id, unique_tables[0], unique_tables[1])
                edges = [(sp[i].split("#sep#")[-1], sp[i + 1].split("#sep#")[-1]) for i in range(len(sp) - 1)]

                edge_joins_map = {}
                for (a, b) in edges:
                    key = f"{a}-{b}"
                    edge_joins_map[key] = get_possible_joins(a, b, db_id)

                results.append({
                    "query": query_text,
                    "db_id": db_id,
                    "optimized_paths": [t.split("#sep#")[-1] for t in sp],
                    "edges_to_check": [edges],
                    "possible_joins_for_edges": edge_joins_map,
                    "subquery_tables": subquery_tables
                })
            else:
                results.append({
                    "query": query_text,
                    "db_id": db_id,
                    "optimized_paths": [[t1], [t2]],
                    "edges_to_check": [],
                    "possible_joins_for_edges": {},
                    "subquery_tables": subquery_tables
                })

        # Case 3: more than two tables
        else:
            components = G.get_connected_components(db_id)
            comp_groups = []
            for comp in components:
                tables_in_query = [t for t in unique_tables if t in comp]
                if tables_in_query:
                    comp_groups.append(tables_in_query)

            subpaths = []
            edges_to_check_grouped = []
            possible_joins_for_edges = {}

            for tables in comp_groups:
                if len(tables) == 1:
                    subpaths.append([tables[0].split("#sep#")[-1]])
                    edges_to_check_grouped.append([tables[0].split("#sep#")[-1]])
                    continue

                subgraph_edges = []
                for i in range(len(tables)):
                    for j in range(i + 1, len(tables)):
                        if G.has_path(db_id, tables[i], tables[j]):
                            path = G.get_shortest_path(db_id, tables[i], tables[j])
                            edges = [(path[k].split("#sep#")[-1], path[k + 1].split("#sep#")[-1]) for k in range(len(path) - 1)]
                            subgraph_edges.extend(edges)


                            for (a, b) in edges:
                                key = f"{a}-{b}"
                                possible_joins_for_edges[key] = get_possible_joins(a, b, db_id)
                        else:
                            subgraph_edges.append((tables[i].split("#sep#")[-1], tables[j].split("#sep#")[-1]))

                subgraph_edges = list(set(subgraph_edges))
                subpaths.append(subgraph_edges)
                edges_to_check_grouped.append(subgraph_edges)

            results.append({
                "query": query_text,
                "db_id": db_id,
                "optimized_paths": table_names,
                "edges_to_check": edges_to_check_grouped,
                "possible_joins_for_edges": possible_joins_for_edges,
                "subquery_tables": subquery_tables
            })

    return results


# def analyze_query_paths2(G, queries):
#     results = []

#     for q in queries:
#         paths = q.get("path", [])
#         if not paths or len(paths[0]) == 0:
#             results.append({
#                 "query": q["query"],
#                 "optimized_paths": [],
#                 "edges_to_check": []
#             })
#             continue


#         raw_tables = [p for p in paths[0]]
#         db_id = raw_tables[0].split("#sep#")[0]
#         unique_tables = list(set(raw_tables))


#         if len(unique_tables) == 1:
#             if len(unique_tables) == 1:
#                 results.append({
#                     "query": q["query"],
#                     "optimized_paths": [[unique_tables[0]]],
#                     "edges_to_check": [[unique_tables[0]]]
#             })
#             continue

#         elif len(unique_tables) == 2:
#             t1, t2 = unique_tables
#             if jg.has_path(db_id, t1, t2):
#                 sp = jg.get_shortest_path(db_id, t1, t2)
#                 edges = [(sp[i], sp[i + 1]) for i in range(len(sp) - 1)]
#                 results.append({
#                     "query": q["query"],
#                     "optimized_paths": sp,
#                     "edges_to_check": [edges]
#                 })
#             else:
#                 results.append({
#                     "query": q["query"],
#                     "optimized_paths": [[t1], [t2]],
#                     "edges_to_check": [[t1], [t2]]
#                 })

#         else:
#             components = jg.get_connected_components(db_id)
#             comp_groups = []
#             for comp in components:
#                 tables_in_query = [t for t in unique_tables if t in comp]
#                 if tables_in_query:
#                     comp_groups.append(tables_in_query)

#             subpaths = []
#             paths = []
#             edges_to_check_grouped = []

#             for tables in comp_groups:
#                 if len(tables) == 1:
#                     subpaths.append([tables[0]])
#                     edges_to_check_grouped.append([tables[0]])
#                 else:
#                     subgraph_edges = []
#                     for i in range(len(tables)):
#                         for j in range(i + 1, len(tables)):
#                             if jg.has_path(db_id, tables[i], tables[j]):
#                                 path = jg.get_shortest_path(db_id, tables[i], tables[j])
#                                 edges = [(path[k], path[k + 1]) for k in range(len(path) - 1)]
#                                 subgraph_edges.extend(edges)
#                             else:
#                                 paths.append((tables[i], tables[j]))
#                                 subgraph_edges.append((tables[i], tables[j]))
#                     subgraph_edges = list(set(subgraph_edges))
#                     subpaths.append(subgraph_edges)
#                     paths.append(path)
#                     edges_to_check_grouped.append(subgraph_edges)

#             results.append({
#                 "query": q["query"],
#                 "optimized_paths": subpaths,
#                 "edges_to_check": edges_to_check_grouped
#             })

#     return results


def main():
    json_file = r"jar2-main\data\bird\possible_table_paths_only_all_sub_queries_included.json"
    # output_file = r"jar2-main\data\bird\analyzed_query_paths.json"
    # jar2-main\data\bird\possible_table_paths_only_all_sub_queries_included.json
    subquery_file = r"jar2-main\data\bird\subquery_to_column_mapping_semantic.json"
    output_file = r"jar2-main\data\bird\analyzed_query_paths3_allsq.json"
    with open(subquery_file, "r", encoding="utf-8") as f:
        subquery_mappings = json.load(f) 
    with open(json_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    results = analyze_query_paths(jg, queries, subquery_mappings)
    for res in results:
        print("\nQuery:", res["query"])
        if "optimized_paths" in res:
            print("Optimized Paths:", res["optimized_paths"])
        print("Edges to Check:", res["edges_to_check"])
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nAnalysis complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()
