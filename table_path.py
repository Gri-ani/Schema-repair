mport json
from typing import List, Dict, Any  # for type hints (optional)

class Table_paths:
    def __init__(self, validated_mappings: List[Dict[str, Any]]):
        self.validated_mappings = validated_mappings

    def generate_paths(self) -> List[Dict[str, Any]]:
        all_query_paths = []

        for query in self.validated_mappings:
            sub_queries = query.get("sub_queries_mapping", [])
            query_text = query.get("query", "<unknown query>")

            # Merge all tables from all sub-queries into a single set
            all_tables = set()
            for sq in sub_queries:
                all_tables.update({tm["table"] for tm in sq.get("top_matches", [])})

            # Convert set to list for JSON serialization
            tables_list = list(all_tables)

            all_query_paths.append({
                "query": query_text,
                "path": tables_list  # single path containing all tables
            })

        return all_query_paths


if __name__ == "__main__":
    input_file = r"./jar2-main/data/bird/subquery_to_column_mapping_semantic.json"

    with open(input_file, "r") as f:
        validated_mappings = json.load(f)

    path_generator = Table_paths(validated_mappings)
    all_paths = path_generator.generate_paths()

    output_file = r"./jar2-main/data/bird/possible_table_paths_only.json"
    with open(output_file, "w") as f:
        json.dump(all_paths, f, indent=2)

    print(f"Generated paths for {len(all_paths)} queries")
    print(f"Output saved to: {output_file}")
