"""Exercise 1: tokenize text and run a pretrained sentiment classifier."""
import argparse
from common import ROOT, configure, save_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", nargs="+", default=["I really enjoyed this movie. The acting was excellent!", "This movie was boring and disappointing."])
    parser.add_argument("--model", default="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    args = parser.parse_args()
    configure()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()
    batch = tokenizer(args.text, padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = model(**batch).logits.softmax(dim=-1)
    results = []
    for i, sentence in enumerate(args.text):
        label_id = int(probs[i].argmax())
        encoded = tokenizer(sentence, truncation=True, max_length=256)
        row = {"text": sentence, "tokens": tokenizer.convert_ids_to_tokens(encoded["input_ids"]),
               "input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
               "label": model.config.id2label[label_id], "score": float(probs[i,label_id])}
        results.append(row)
        print(f"{row['label']} ({row['score']:.4f}): {sentence}")
        print("Tokens:", row["tokens"])
    save_json(ROOT/"outputs/exercise1/results.json", {"model": args.model, "results": results})
    print("Saved: outputs/exercise1/results.json")

if __name__ == "__main__":
    main()
