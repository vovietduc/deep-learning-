"""Practice 3 - Hugging Face. Chạy: python main.py --exercise all."""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Practice 3: sentiment analysis và fine-tuning")
    parser.add_argument("--exercise", choices=["1", "2", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--full-data", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs và batch-size phải lớn hơn 0")
    return args

# Parse trước khi import thư viện ML để --help dùng được ngay.
if __name__ == "__main__":
    args = parse_args()

from check_env import check_environment
if not check_environment():
    raise SystemExit(1)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import gc
from pathlib import Path
import numpy as np
import torch
import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, set_seed,
)
SEED = 42
set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Thiết bị:", DEVICE)
if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Đang dùng CPU; fine-tuning có thể chậm. GPU NVIDIA phù hợp sẽ giúp chạy nhanh hơn.")
OUTPUT_DIR = Path(__file__).resolve().parent / "practice3_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def exercise1():
    SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer_ex1 = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    model_ex1 = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL).to(DEVICE)
    model_ex1.eval()
    print("Ánh xạ nhãn:", model_ex1.config.id2label)

    sample_sentence = "I really enjoyed this movie. The acting was excellent!"
    encoded = tokenizer_ex1(sample_sentence, return_tensors="pt", truncation=True, max_length=256)
    print("Câu mẫu:", sample_sentence)
    print("Tokens:", tokenizer_ex1.convert_ids_to_tokens(encoded["input_ids"][0].tolist()))
    print("input_ids:", encoded["input_ids"])
    print("attention_mask:", encoded["attention_mask"])

    with torch.no_grad():
        logits = model_ex1(**{k: v.to(DEVICE) for k, v in encoded.items()}).logits
        probabilities = torch.softmax(logits, dim=-1)[0].cpu()
    predicted_id = int(probabilities.argmax())
    print("Kết quả:", model_ex1.config.id2label[predicted_id])
    print("Xác suất lớp dự đoán:", round(float(probabilities[predicted_id]), 4))
    for class_id, probability in enumerate(probabilities):
        print(model_ex1.config.id2label[class_id], "=", round(float(probability), 4))

    examples = [
        sample_sentence,
        "This movie was boring and the acting was terrible.",
        "The story was interesting, but the ending was disappointing.",
    ]
    inputs = tokenizer_ex1(examples, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        scores = model_ex1(**{k: v.to(DEVICE) for k, v in inputs.items()}).logits.softmax(-1).cpu()
    exercise1_results = []
    for text, score in zip(examples, scores):
        idx = int(score.argmax())
        row = {"text": text, "label": model_ex1.config.id2label[idx], "score": float(score[idx])}
        exercise1_results.append(row)
        print(row)
    (OUTPUT_DIR / "exercise1_results.json").write_text(json.dumps(exercise1_results, indent=2, ensure_ascii=False), encoding="utf-8")
    # Giải phóng mô hình bài 1 trước khi fine-tune bài 2.
    del model_ex1, inputs, encoded, logits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def exercise2(full_data=False, epochs=2, batch_size=8):
    FULL_DATA = full_data
    raw = load_dataset("stanfordnlp/imdb")
    print(raw)

    def stratified_subset(dataset, size):
        if size >= len(dataset):
            return dataset.shuffle(seed=SEED)
        return dataset.train_test_split(train_size=size, stratify_by_column="label", seed=SEED)["train"]

    pool = raw["train"] if FULL_DATA else stratified_subset(raw["train"], 2500)
    split = pool.train_test_split(test_size=0.2, stratify_by_column="label", seed=SEED)
    dataset = DatasetDict({
        "train": split["train"],
        "validation": split["test"],
        "test": raw["test"] if FULL_DATA else stratified_subset(raw["test"], 500),
    })
    for name, part in dataset.items():
        labels, counts = np.unique(part["label"], return_counts=True)
        print(name, "số mẫu:", len(part), "phân bố nhãn:", dict(zip(labels.tolist(), counts.tolist())))
    print("Ví dụ:", dataset["train"][0]["text"][:400])
    print("Nhãn:", dataset["train"][0]["label"])

    BASE_MODEL = "distilbert/distilbert-base-uncased"
    MAX_LENGTH = 256
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {name: idx for idx, name in id2label.items()}
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(preprocess, batched=True, remove_columns=["text"])
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label=id2label, label2id=label2id,
    )
    print("Tổng tham số:", sum(p.numel() for p in model.parameters()))
    print("Tham số được học:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    try:
        accuracy_metric = evaluate.load("accuracy")
    except (OSError, ConnectionError, FileNotFoundError) as exc:
        print("Không tải được metric accuracy; dùng sklearn:", type(exc).__name__)
        accuracy_metric = None

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
                    if accuracy_metric is not None else accuracy_score(labels, predictions))
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=1, zero_division=0,
        )
        return {"accuracy": float(accuracy), "precision": float(precision),
                "recall": float(recall), "f1": float(f1)}

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / "checkpoints"),
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=0.1,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        report_to="none",
        push_to_hub=False,
        seed=SEED,
        data_seed=SEED,
        fp16=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    print("Checkpoint tốt nhất:", trainer.state.best_model_checkpoint)

    test_output = trainer.predict(tokenized["test"], metric_key_prefix="test")
    trainer.log_metrics("test", test_output.metrics)
    trainer.save_metrics("test", test_output.metrics)
    y_true = test_output.label_ids
    y_pred = np.argmax(test_output.predictions, axis=-1)
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["NEGATIVE", "POSITIVE"], digits=4, zero_division=0)
    print(report)
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0, 1], display_labels=["NEGATIVE", "POSITIVE"], cmap="Blues")
    plt.title("Confusion matrix — IMDb test")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Kết quả: Accuracy={test_output.metrics['test_accuracy']:.4f}; F1={test_output.metrics['test_f1']:.4f}")

    history = trainer.state.log_history
    train_logs = [row for row in history if "loss" in row]
    val_logs = [row for row in history if "eval_loss" in row]
    plt.figure(figsize=(8, 4))
    if train_logs:
        plt.plot([r["epoch"] for r in train_logs], [r["loss"] for r in train_logs], label="Train loss (logged average)")
    if val_logs:
        plt.plot([r["epoch"] for r in val_logs], [r["eval_loss"] for r in val_logs], "o-", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=150)
    plt.close()

    MODEL_DIR = OUTPUT_DIR / "finetuned_distilbert_imdb"
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))
    # Nạp bản lưu lên CPU để không giữ thêm một bản model trên GPU.
    loaded_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    loaded_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR)).eval()
    new_texts = ["A wonderful movie with a touching story.", "I regret watching this boring film."]
    new_inputs = loaded_tokenizer(new_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    with torch.no_grad():
        new_probs = loaded_model(**new_inputs).logits.softmax(dim=-1)
    for text, probs in zip(new_texts, new_probs):
        index = int(probs.argmax())
        print({"text": text, "label": loaded_model.config.id2label[index], "score": round(float(probs[index]), 4)})
    print("Đã lưu mô hình tại:", MODEL_DIR.resolve())

if __name__ == "__main__":
    try:
        if args.exercise in ("1", "all"):
            exercise1()
        if args.exercise in ("2", "all"):
            exercise2(args.full_data, args.epochs, args.batch_size)
        print("Hoàn tất. Kết quả được lưu tại:", OUTPUT_DIR)
    except Exception:
        import traceback
        log = OUTPUT_DIR / "error.log"
        log.write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
        print("Đã lưu chi tiết lỗi:", log)
        print("Nếu lỗi tải Hugging Face: kiểm tra kết nối Internet. Nếu thiếu thư viện: chạy lại setup.")
        raise SystemExit(1)
