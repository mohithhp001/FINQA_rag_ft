# src/train/ft.py
import argparse, pandas as pd, sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from datasets import Dataset

try:
    from peft import LoraConfig, get_peft_model
    PEFT = True
except Exception:
    PEFT = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='CSV with columns: question,answer')
    ap.add_argument('--out', default='models/ft-flan-t5-small')
    ap.add_argument('--base', default='google/flan-t5-small')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bsz', type=int, default=8)
    ap.add_argument('--lr', type=float, default=5e-4)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if not {'question','answer'}.issubset(df.columns):
        raise ValueError('CSV must have columns: question,answer')

    # Make everything strings and drop empties
    df['question'] = df['question'].astype(str).str.strip()
    df['answer'] = df['answer'].fillna('').astype(str).str.strip()
    df = df[(df['question']!='') & (df['answer']!='')].reset_index(drop=True)
    if len(df) == 0:
        print("No non-empty answers to train on. Fill answers first.", file=sys.stderr)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base)

    if PEFT:
        lora = LoraConfig(
            r=8, lora_alpha=16, target_modules=["q","v","k","o"],
            lora_dropout=0.05, bias="none", task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, lora)

    def prep(batch):
        inputs = ["question: " + q for q in batch['question']]
        model_inputs = tok(inputs, truncation=True)
        labels = tok(text_target=batch['answer'], truncation=True)  # <- key change
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    ds = Dataset.from_pandas(df[['question','answer']])
    ds = ds.map(prep, batched=True, remove_columns=['question','answer'])

    collator = DataCollatorForSeq2Seq(tok, model=model)
    args_tr = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bsz,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False, bf16=False,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args_tr, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"Saved fine-tuned model to {args.out}")

if __name__ == "__main__":
    main()
