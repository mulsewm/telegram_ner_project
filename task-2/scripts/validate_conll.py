def validate_conll(file_path):
    """
    Validates the CoNLL format of the labeled data file.
    """
    errors = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            if line.strip() and len(line.split()) != 2:
                print(f"Error on line {i}: {line.strip()}")
                errors += 1
    if errors == 0:
        print("No formatting errors found!")
    else:
        print(f"Validation completed with {errors} errors.")

if __name__ == "__main__":
    file_path = "task-2/data/labeled_data.conll"
    validate_conll(file_path)
