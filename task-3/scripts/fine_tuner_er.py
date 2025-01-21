from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import load_dataset, Dataset
import os
import json

def prepare_dataset(tokenized_file, model_name="xlm-roberta-base"):
    # Define the complete label map based on your dataset
    label_map = {
        "O": 0,          # Outside any entity
        "B-LOC": 1,      # Beginning of location
        "I-LOC": 2,      # Inside location
        "B-PRODUCT": 3,  # Beginning of product
        "I-PRODUCT": 4,  # Inside product
        "B-PRICE": 5,    # Beginning of price
        "I-PRICE": 6,    # Inside price
        # Add any other labels you have in your dataset
    }

    with open(tokenized_file, 'r', encoding='utf-8') as f:
        data = []
        for line in f:
            try:
                line = line.replace("'", '"')
                entry = json.loads(line.strip())
                entry["labels"] = [label_map.get(label, -1) for label in entry["labels"]]

                # Skip entries with invalid labels
                if -1 in entry["labels"]:
                    print(f"Invalid label found in entry: {entry['labels']}")
                    continue
                
                data.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line: {line.strip()}")
                print(f"Error message: {e}")
                continue

    print(f"Total valid entries: {len(data)}")

    if len(data) == 0:
        raise ValueError("The dataset contains no valid entries after label mapping.")

    # Split into train and validation sets
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)

    if len(train_dataset) == 0:
        raise ValueError("The training dataset is empty. Please check your data and label mapping.")

    return train_dataset, val_dataset


def fine_tune_model(train_dataset, val_dataset, model_name="xlm-roberta-base", output_dir="task-3/models/ner"):
    """
    Fine-tune the pre-trained model for NER.
    :param train_dataset: Training dataset.
    :param val_dataset: Validation dataset.
    :param model_name: Name of the pre-trained model.
    :param output_dir: Path to save the fine-tuned model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=5)  # Adjust num_labels as needed

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        save_steps=500,
        eval_steps=500,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        logging_dir="task-3/logs",
        logging_steps=100,
        save_total_limit=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        report_to="none",  # Disable reporting to external tools
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model saved to {output_dir}")


if __name__ == "__main__":
    tokenized_file = "task-3/data/tokenized_data.json"
    output_dir = "task-3/models/ner"
    model_name = "xlm-roberta-base"

    os.makedirs(output_dir, exist_ok=True)

    # Prepare the dataset
    train_dataset, val_dataset = prepare_dataset(tokenized_file, model_name)

    # Fine-tune the model
    fine_tune_model(train_dataset, val_dataset, model_name, output_dir)
