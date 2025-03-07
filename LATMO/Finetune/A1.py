import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch.cuda as cuda
from huggingface_hub import login

# Set your Hugging Face token
HF_TOKEN = os.getenv('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')
login(token=HF_TOKEN)  # Login to Hugging Face

# Check CUDA availability
print(f"CUDA available: {cuda.is_available()}")
if cuda.is_available():
    print(f"GPU Model: {cuda.get_device_name(0)}")
    print(f"Available VRAM: {cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Configuration
MODEL_NAME = "google/gemma-2-2b-it"  # You can change this to any model from HuggingFace
DATASET_NAME = "AXEUS/Name-LT"  # Change this to your desired dataset
NUM_LABELS = 2  # Change based on your task
BATCH_SIZE = 4  # Small batch size due to VRAM constraints
ACCUMULATION_STEPS = 4  # Effective batch size will be BATCH_SIZE * ACCUMULATION_STEPS
EPOCHS = 3
LEARNING_RATE = 5e-5

print(f"Loading dataset: {DATASET_NAME}")
try:
    dataset = load_dataset(DATASET_NAME)
except Exception as e:
    raise Exception(f"Failed to load dataset {DATASET_NAME}: {str(e)}")

print(f"Loading tokenizer: {MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN,
        trust_remote_code=True
    )
except Exception as e:
    raise Exception(f"Failed to load tokenizer: {str(e)}")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )

print("Tokenizing datasets...")
try:
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
except Exception as e:
    raise Exception(f"Failed to tokenize datasets: {str(e)}")

print("Loading model...")
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        torch_dtype=torch.float16,
        token=HF_TOKEN,
        trust_remote_code=True
    )
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")

# Configure LoRA for parameter-efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
)
model = get_peft_model(model, peft_config)

# Training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    gradient_accumulation_steps=ACCUMULATION_STEPS,
    fp16=True,  # Mixed precision training
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_steps=10,  # Add more frequent logging
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),  # Add logging directory
)

print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

print("Starting training...")
try:
    trainer.train()
except Exception as e:
    raise Exception(f"Training failed: {str(e)}")

print("Saving model...")
try:
    trainer.save_model(OUTPUT_DIR)
    print(f"Model successfully saved to {OUTPUT_DIR}")
except Exception as e:
    raise Exception(f"Failed to save model: {str(e)}")
