from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

def test_model(model_dir, sentence):
    """
    Load the fine-tuned model and test it on a sample sentence.
    :param model_dir: Path to the fine-tuned model.
    :param sentence: Sentence to test the model on.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)

    # Tokenize the input sentence
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=True)

    # Perform inference
    with torch.no_grad():
        outputs = model(**tokens)
        predictions = torch.argmax(outputs.logits, dim=2)

    # Map predictions to labels
    labels = model.config.id2label
    tokens = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    predicted_labels = [labels[pred.item()] for pred in predictions[0]]

    # Print results
    for token, label in zip(tokens, predicted_labels):
        print(f"{token}: {label}")


if __name__ == "__main__":
    model_dir = "task-3/models/ner"
    sentence = ["በአዲስ", "አበባ", "ሆስፒታል", "ዋጋ", "1000", "ብር"]
    test_model(model_dir, sentence)
