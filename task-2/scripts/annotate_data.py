def create_labeled_file(raw_file, output_file):
    """
    Simulates the annotation process for demonstration.
    Reads raw data, allows manual annotation, and saves it in CoNLL format.
    """
    with open(raw_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            sentence = line.strip()
            if sentence:
                tokens = sentence.split()
                for token in tokens:
                    # Replace 'O' with an actual label like B-LOC, I-PRODUCT, etc., during annotation
                    label = input(f"Enter label for '{token}' (e.g., O, B-LOC, I-LOC): ") or "O"
                    outfile.write(f"{token}\t{label}\n")
                outfile.write("\n")  # Blank line to separate sentences

if __name__ == "__main__":
    raw_file = "task-2/data/raw_data.txt"
    output_file = "task-2/data/labeled_data.conll"
    create_labeled_file(raw_file, output_file)
