"""Predict with the saved fine-tuned model; no retraining required."""
import argparse
from common import ROOT, configure

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--text",required=True)
    p.add_argument("--model-dir",default=str(ROOT/"outputs/exercise2/model"))
    a=p.parse_args()
    from pathlib import Path
    if not (Path(a.model_dir)/"config.json").is_file():
        p.error("Model not found. Run src/exercise2.py first.")
    configure()
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    tokenizer=AutoTokenizer.from_pretrained(a.model_dir,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(a.model_dir,local_files_only=True)
    model.eval()
    with torch.no_grad():
        probs=model(**tokenizer(a.text,return_tensors="pt",truncation=True,max_length=128)).logits.softmax(-1)[0]
    label=int(probs.argmax())
    print(model.config.id2label[label],f"score={float(probs[label]):.4f}")

if __name__ == "__main__":
    main()
