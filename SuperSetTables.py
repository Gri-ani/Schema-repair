import json
import os
from itertools import combinations
from join_graph2 import JoinabilityGraphs  


class SuperSetTables:
    def __init__(self, join_graphs):
        self.join_graphs = join_graphs

    def get_db_id(self, table_name):
        return table_name.split("#sep#")[0]

    def expand_with_paths(self, table_sets):
        expanded_sets = []

        for group in table_sets:
            if not group:
                expanded_sets.append([])
                continue    
            db_id = self.get_db_id(group[0])
            tables = set(group)
            for t1, t2 in combinations(group, 2):
                if self.join_graphs.has_path(db_id, t1, t2):
                    path = self.join_graphs.get_shortest_path(db_id, t1, t2)
                    if path:
                        tables.update(path)
            expanded_sets.append(sorted(tables))
        return expanded_sets


def main():
    join_graphs = JoinabilityGraphs("bird")
    superset = SuperSetTables(join_graphs)
    input_path = "./jar2-main/data/ilp_preds/contriever/bird_0._check_best.json"
    output_path = "./jar2-main/data/ilp_preds/contriever/bird_0_expanded.json"
    with open(input_path, 'r') as f:
        table_sets = json.load(f)
    expanded = superset.expand_with_paths(table_sets)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(expanded, f, indent=2)
    print(f"Expanded table sets saved to {output_path}")

if __name__ == "__main__":
    main()
