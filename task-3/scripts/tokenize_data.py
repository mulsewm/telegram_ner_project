from transformers import AutoTokenizer
import os

def tokenize_and_align_labels(dataset_path, output_path, model_name="xlm-roberta-base"):
    """
    Tokenizes the dataset and aligns entity labels with tokens.
    :param dataset_path: Path to the CoNLL dataset.
    :param output_path: Path to save the tokenized dataset.
    :param model_name: Name of the pre-trained model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_sentences = []
    labels = []

    with open(dataset_path, "r", encoding="utf-8") as file:
        sentence_tokens = []
        sentence_labels = []
        for line in file:
            line = line.strip()
            if line == "":
                # Process the sentence
                tokenized_sentence = tokenizer(sentence_tokens, is_split_into_words=True, truncation=True, padding=True)
                tokenized_sentence["labels"] = sentence_labels
                tokenized_sentences.append(tokenized_sentence)
                sentence_tokens = []
                sentence_labels = []
            else:
                token, label = line.split()
                sentence_tokens.append(token)
                sentence_labels.append(label)

    # Save tokenized dataset
    with open(output_path, "w", encoding="utf-8") as outfile:
        for sentence in tokenized_sentences:
            outfile.write(f"{sentence}\n")

    print(f"Tokenized data saved to {output_path}")


if __name__ == "__main__":
    dataset_path = "/Users/mulsewsmba/telegram_ner_project/task-2/data/labeled_data.conll"
    output_path = "task-3/data/tokenized_data.json"
    model_name = "xlm-roberta-base"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenize_and_align_labels(dataset_path, output_path, model_name)
