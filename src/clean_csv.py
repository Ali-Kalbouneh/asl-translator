import csv

input_path = "../data/letters.csv"
output_path = "../data/letters_cleaned.csv"

label_to_delete = ""   # <<< change this

rows = []

# Read and filter out unwanted rows
with open(input_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["label"] != label_to_delete:
            rows.append(row)

# Write filtered rows into a new CSV
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Removed all '{label_to_delete}' samples.")
