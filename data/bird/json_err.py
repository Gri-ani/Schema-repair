import json

file_path = r"C:\Users\lalit\Downloads\jar2-main 2 (1)\jar2-main\data\bird\decomp.json"

try:
    with open(file_path, "r") as f:
        data = json.load(f)
    print("Valid JSON ✅")
except json.JSONDecodeError as e:
    print("Invalid JSON ❌")
    print(e)
    # Show the line causing the issue
    with open(file_path, "r") as f:
        lines = f.readlines()
        error_line = lines[e.lineno - 1]  # line numbers start at 1
        print(f"Error at line {e.lineno}: {error_line.strip()}")
