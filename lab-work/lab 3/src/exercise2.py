"""Exercise 2: actual pretrained BERT fine-tuning with Hugging Face Trainer."""
import argparse
import csv
import math
import platform
import time
from common import ROOT, configure, metrics, save_json

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--learning-rate",type=float,default=5e-5)
    parser.add_argument("--max-length",type=int,default=128)
    parser.add_argument("--model",default="prajjwal1/bert-tiny")
    parser.add_argument("--tokenizer",default="google-bert/bert-base-uncased")
    parser.add_argument("--output-dir",default=str(ROOT/"outputs/exercise2"))
    parser.add_argument("--cpu",action="store_true",help="Force CPU even if CUDA is available")
    parser.add_argument("--smoke",action="store_true",help="Only 2 training steps; not a quality benchmark")
    args=parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    configure()
    from pathlib import Path
    import numpy as np
    import torch, transformers, datasets
    from datasets import load_dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
        TrainingArguments, Trainer, BertConfig, AutoConfig)
    if not all((ROOT/"data"/(s+".csv")).exists() for s in ["train","validation","test"]):
        from prepare_data import prepare
        prepare()
    data=load_dataset("csv",data_files={s:str(ROOT/"data"/(s+".csv")) for s in ["train","validation","test"]})
    if args.smoke:
        for s in data:
            data[s]=data[s].select(range(min(16,len(data[s]))))
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer)
    encoded=data.map(lambda batch:tokenizer(batch["text"],truncation=True,max_length=args.max_length),
                     batched=True,remove_columns=["text"])
    # The original bert-tiny config omits model_type; specify BertConfig explicitly.
    config_class = BertConfig if args.model == "prajjwal1/bert-tiny" else AutoConfig
    config = config_class.from_pretrained(args.model, num_labels=2,
        id2label={0:"NEGATIVE",1:"POSITIVE"}, label2id={"NEGATIVE":0,"POSITIVE":1})
    model=AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    total_steps = 2 if args.smoke else math.ceil(len(encoded["train"])/args.batch_size)*args.epochs
    training_args=TrainingArguments(output_dir=str(out/"checkpoints"),
        num_train_epochs=args.epochs, max_steps=2 if args.smoke else -1,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, weight_decay=0.01,
        warmup_steps=math.ceil(total_steps*0.1), lr_scheduler_type="linear", max_grad_norm=1.0,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=10,
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        save_total_limit=1, report_to="none", optim="adamw_torch", seed=42,
        data_seed=42, dataloader_num_workers=0, dataloader_pin_memory=False,
        use_cpu=args.cpu or not torch.cuda.is_available(), fp16=False,bf16=False)
    trainer=Trainer(model=model,args=training_args,train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),compute_metrics=metrics)
    start=time.perf_counter()
    training_result=trainer.train()
    validation=trainer.evaluate()
    prediction=trainer.predict(encoded["test"],metric_key_prefix="test")
    trainer.save_model(str(out/"model"));tokenizer.save_pretrained(out/"model")
    pred=np.argmax(prediction.predictions,axis=-1)
    confusion=[[int(((np.asarray(data["test"]["label"])==i)&(pred==j)).sum()) for j in [0,1]] for i in [0,1]]
    with (out/"test_predictions.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f);writer.writerow(["text","true_label","predicted_label"])
        writer.writerows((row["text"],row["label"],int(y)) for row,y in zip(data["test"],pred))
    result={"model":args.model,"smoke_test":args.smoke,"seed":42,
        "python":platform.python_version(),"platform":platform.platform(),
        "versions":{"torch":torch.__version__,"transformers":transformers.__version__,"datasets":datasets.__version__},
        "device":str(training_args.device),"sizes":{s:len(data[s]) for s in data},
        "parameters":sum(p.numel() for p in model.parameters()),
        "hyperparameters":vars(args),"training":training_result.metrics,
        "validation":validation,"test":prediction.metrics,"confusion_matrix":confusion,
        "elapsed_seconds":time.perf_counter()-start}
    save_json(out/"metrics.json",result);save_json(out/"training_history.json",trainer.state.log_history)
    print("\nTEST RESULTS:",prediction.metrics)
    print("Confusion matrix (rows=true, columns=predicted; 0=negative, 1=positive):",confusion)
    print("Saved model and results to",out)

if __name__ == "__main__":
    main()
