import torch
from transformers import GPT2Tokenizer, GPT2Config, AdamW, GPT2LMHeadModel
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import random
from data import get_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]['text']
        label = self.texts[idx].get('label', 0)
        encodings = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'labels': input_ids, 
            'binary_label': torch.tensor(label, dtype=torch.float)
        }

class GPT2WithFreshClassifier(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # Define the classifier architecture to take in 16 floats
        self.classifier = nn.Sequential(
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )

    def forward(self, input_ids, attention_mask=None, labels=None, binary_labels=None):
        outputs = super().forward(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            output_hidden_states=True
        )
        lm_loss = outputs.loss

        logits = outputs.logits[:, :16, :]  # Take only the first 16 positions
        # Calculate the sum of squares of softmaxed logits at each of the 16 positions
        logits_softmax = F.softmax(logits, dim=-1)
        sum_of_squares = torch.sum(logits_softmax ** 2, dim=-1)

        # Pass the sum of squares to the classifier
        cls_logits = self.classifier(sum_of_squares)

        binary_loss = None
        if binary_labels is not None:
            binary_loss = F.binary_cross_entropy_with_logits(cls_logits.squeeze(), binary_labels)

        return {
            'lm_loss': lm_loss, 
            'binary_loss': binary_loss,
            'logits': logits, 
            'cls_logits': cls_logits
        }

# Load the GPT-2 model configuration and instantiate your custom model as before
config = GPT2Config.from_pretrained('gpt2')
model = GPT2WithFreshClassifier(config)
model.transformer = GPT2LMHeadModel.from_pretrained('./model_files/less_overfit_model').transformer


# Resize the token embeddings
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

# Freeze GPT-2 parameters
for param in model.parameters():
    param.requires_grad = False

# Unfreeze classifier parameters
for param in model.classifier.parameters():
    param.requires_grad = True

model = model.to(device)

# Load training data
train_data = get_data(data_type='train', n=2_000, duplication_level=1)
reserve_data = get_data(data_type='reserve', n=2_000)

# Combine and shuffle the data
holdout_data = train_data[:100] + reserve_data[:100]
combined_data = train_data[100:] + reserve_data[100:]
random.shuffle(combined_data)
random.shuffle(holdout_data)

train_dataset = TextDataset(combined_data, tokenizer, 256)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)

# Set up optimizer
optimizer_cls = AdamW(model.classifier.parameters(), lr=5e-3)

# Training loop with updates every k batches
k = 100  # Adjust this value as needed
model.train()
for epoch in range(1):
    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        binary_labels = batch['binary_label'].to(device)
        # print(binary_labels)

        optimizer_cls.zero_grad()

        outputs = model(input_ids, attention_mask=attention_mask, binary_labels=binary_labels)
        binary_loss = outputs['binary_loss']

        binary_loss.backward()
        optimizer_cls.step()

        # Print loss every k batches
        if (batch_idx + 1) % k == 0:
            print(f'Batch {batch_idx + 1}/{len(train_dataloader)} - Binary Loss: {binary_loss.item()}')

    print(f'Epoch {epoch + 1}/{1} - Final Binary Loss: {binary_loss.item()}')


test_dataset = TextDataset(holdout_data, tokenizer, 256)
test_dataloader = DataLoader(test_dataset, batch_size=48, shuffle=False)

# Switch to evaluation mode
model.eval()

# Initialize variables to track the loss and accuracy
total_loss = 0.0
correct_predictions = 0
total_predictions = 0

with torch.no_grad():
    for batch in tqdm(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        binary_labels = batch['binary_label'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, binary_labels=binary_labels)
        binary_loss = outputs['binary_loss']
        cls_logits = outputs['cls_logits']

        # Accumulate the loss
        total_loss += binary_loss.item()

        # Calculate the number of correct predictions
        predictions = (cls_logits.squeeze() > 0).float()
        print(predictions)
        correct_predictions += (predictions == binary_labels).sum().item()
        total_predictions += binary_labels.size(0)

# Calculate the average loss and accuracy
average_loss = total_loss / len(test_dataloader)
accuracy = correct_predictions / total_predictions

print(f'Final Evaluation - Average Binary Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}')

