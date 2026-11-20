"""Export deterministic subsets of official Rotten Tomatoes splits to CSV."""
import argparse
import csv
from common import ROOT, save_json

def prepare(train_size=1000, validation_size=200, test_size=200, seed=42):
    from datasets import load_dataset
    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
    provenance = {"source": "cornell-movie-review-data/rotten_tomatoes", "seed": seed,
                  "selection": "shuffle(seed) then select; original train/validation/test splits retained", "splits": {}}
    for name, size in [("train",train_size),("validation",validation_size),("test",test_size)]:
        part = dataset[name].shuffle(seed=seed).select(range(min(size,len(dataset[name]))))
        path = ROOT/"data"/(name+".csv")
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("w",encoding="utf-8",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["text","label"])
            writer.writeheader(); writer.writerows(part)
        provenance["splits"][name] = {"size": len(part), "negative": part["label"].count(0), "positive": part["label"].count(1), "fingerprint": part._fingerprint}
    save_json(ROOT/"data/provenance.json",provenance)
    print("Saved train.csv, validation.csv, test.csv and provenance.json")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-size",type=int,default=1000)
    parser.add_argument("--validation-size",type=int,default=200)
    parser.add_argument("--test-size",type=int,default=200)
    a=parser.parse_args()
    prepare(a.train_size,a.validation_size,a.test_size)
