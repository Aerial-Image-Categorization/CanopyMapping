from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, TrainingArguments, Trainer
import torch
import numpy as np

# Load the dataset (or use a custom dataset)
dataset = load_dataset("mbnczy/bakonyszucs_200")

# Load the image processor and model
image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Preprocess images and masks
def preprocess_images_and_masks(examples):
    inputs = image_processor(examples['image'], return_tensors="pt")
    labels = np.array(examples['label'], dtype=np.int64)
    labels = torch.tensor(labels, dtype=torch.long)
    inputs["labels"] = labels
    return inputs

# Apply preprocessing
train_dataset = dataset["train"].map(preprocess_images_and_masks, batched=True)
val_dataset = dataset["validation"].map(preprocess_images_and_masks, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    save_total_limit=3,
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train the model
trainer.train()

# Evaluate the model
metrics = trainer.evaluate()
print(metrics)

# Save the trained model
trainer.save_model("./segformer-finetuned")
