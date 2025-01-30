import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
from deprecated_data import get_data
import torch.nn.functional as F

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]['text']
        label = self.texts[idx].get('label', 0)  # Use .get() to avoid KeyError
        encodings = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encodings['input_ids'].squeeze()
        attention_mask = encodings['attention_mask'].squeeze()
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids, 'binary_label': torch.tensor(label, dtype=torch.float)}

class GPT2WithBinaryClassifier(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.classifier = torch.nn.Linear(config.n_embd, 1)

    # def forward(self, input_ids, attention_mask=None, labels=None, binary_labels=None):
    #     outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
    #     logits = outputs.logits
    #     hidden_states = outputs.hidden_states[-1]  # Get the last hidden states
    #     cls_logits = self.classifier(hidden_states[:, 0, :])  # Take the output for the [CLS] token
        
    #     loss = outputs.loss
    #     if binary_labels is not None:
    #         binary_loss = F.binary_cross_entropy_with_logits(cls_logits.squeeze(), binary_labels)
    #         loss = loss + binary_loss

    #     return {'loss': loss, 'logits': logits, 'cls_logits': cls_logits}
    def forward(self, input_ids, attention_mask=None, labels=None, binary_labels=None):
        # Forward pass through the GPT-2 model, getting hidden states
        outputs = super().forward(input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden states

        # Determine the index of the last non-padding token for each sequence in the batch
        last_token_indices = attention_mask.sum(dim=1) - 1  # Find the last non-padding token index

        # Use advanced indexing to select the hidden state for the last non-padding token
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)  # Create batch indices
        last_hidden_states = hidden_states[batch_indices, last_token_indices, :]  # Select last non-padding token's hidden state
        cls_logits = self.classifier(last_hidden_states)  # Apply classifier to the selected hidden states
        
        # Compute the standard language model loss
        loss = outputs.loss
        if binary_labels is not None:
            # Compute the binary classification loss
            binary_loss = F.binary_cross_entropy_with_logits(cls_logits.squeeze(), binary_labels)
            loss = loss + binary_loss  # Combine the losses

        return {'loss': loss, 'logits': logits, 'cls_logits': cls_logits}

def train_batch(model, batch, optimizer, scheduler, device, alpha, report=False):
    model.train()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    binary_labels = batch['binary_label'].to(device)

    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels, binary_labels=binary_labels)
    loss = alpha * outputs['loss'] + (1 - alpha) * F.binary_cross_entropy_with_logits(outputs['cls_logits'].squeeze(), binary_labels)
    if report:
        print('train stuff:')
        print(outputs['loss'], F.binary_cross_entropy_with_logits(outputs['cls_logits'].squeeze(), binary_labels))
    loss.backward()
    optimizer.step()
    scheduler.step()

def evaluate(model, dataloader, device, include_binary_loss=True):
    model.eval()
    total_loss = 0
    total_binary_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            binary_labels = batch['binary_label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, binary_labels=binary_labels)
            total_loss += outputs['loss'].item()
            if include_binary_loss:
                total_binary_loss += F.binary_cross_entropy_with_logits(outputs['cls_logits'].squeeze(), binary_labels).item()

    if include_binary_loss:
        return total_loss / len(dataloader), total_binary_loss / len(dataloader)
    else:
        return total_loss / len(dataloader), 0

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load training data
    train_dataset = get_data(args.num_train_examples, max_inventory=args.num_train_examples // 100, include_labels=True, zeros=False, ratio=0.5, offline=True, batch_size=args.batch_size)

    # Load test data
    test_dataset = get_data(args.num_test_examples, start_index=args.num_train_examples, include_labels=False, zeros=True, offline=True)
    
    # Create PyTorch datasets
    train_dataset = TextDataset(train_dataset, tokenizer, args.max_length)
    test_dataset = TextDataset(test_dataset, tokenizer, args.max_length)
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Initialize the GPT-2 model with random weights
    config = GPT2Config(pad_token_id=tokenizer.pad_token_id, output_hidden_states=True)
    model = GPT2WithBinaryClassifier(config)
    model.resize_token_embeddings(len(tokenizer))  # Resize the token embeddings to accommodate the new pad token
    model = model.to(device)

    # Set up the optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Train the model and evaluate at specified intervals within the single epoch
    for step, batch in enumerate(tqdm(train_dataloader, total=total_steps)):
        report = False
        if step % (total_steps // args.num_evals_per_epoch) == 0:
            val_loss, val_binary_loss = evaluate(model, test_dataloader, device, include_binary_loss=False)
            print(f'Step {step}/{total_steps} - Validation Loss: {val_loss}, Validation Binary Loss: {val_binary_loss}')
            report = True
        train_batch(model, batch, optimizer, scheduler, device, args.alpha, report=report)
    
    # Save the model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, default=100_000, help='Number of training examples')
    parser.add_argument('--num_test_examples', type=int, default=2000, help='Number of testing examples')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='./model_files/gpt2_model', help='Directory to save the model')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for the NTP loss')
    parser.add_argument('--num_evals_per_epoch', type=int, default=20, help='Number of evaluations per epoch')
    args = parser.parse_args()
    
    main(args)
