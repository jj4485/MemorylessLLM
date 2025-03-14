import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
from deprecated_data import get_data
import torch.nn.functional as F
import torch.nn as nn
import math

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

class GPT2WithBinaryClassifier(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        # Define a small MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 16),  # First hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(config.n_embd // 16, config.n_embd // 32),  # Second hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(config.n_embd // 32, 1)  # Output layer
        )

    def forward(self, input_ids, attention_mask=None, labels=None, binary_labels=None, ratio=0.5):
        # Forward pass through the GPT-2 model, getting hidden states
        outputs = super().forward(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            output_hidden_states=True
        )
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # Get the last layer's hidden states

        # Determine the index of the last non-padding token for each sequence in the batch
        last_token_indices = attention_mask.sum(dim=1) - 1  # Find the last non-padding token index

        # Use advanced indexing to select the hidden state for the last non-padding token
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)  # Create batch indices
        last_hidden_states = hidden_states[batch_indices, last_token_indices, :]  # Select last non-padding token's hidden state
        
        # Compute classification logits using the same classifier for both losses
        cls_logits = self.classifier(last_hidden_states)  # Apply classifier to the last hidden states
        
        # Compute the standard language model loss
        lm_loss = outputs.loss

        # Compute binary classification loss with real labels (only updates classifier)
        binary_loss = None
        if binary_labels is not None:
            # Detach last_hidden_states to prevent gradient flow through LM
            cls_logits_detached = self.classifier(last_hidden_states.detach())
            binary_loss = F.binary_cross_entropy_with_logits(cls_logits_detached.squeeze(), binary_labels)

        # Compute adversarial binary prediction loss with constant labels (only updates LM)
        adversarial_binary_loss = None
        if binary_labels is not None:
            # Use detached classifier logits for adversarial binary loss
            adversarial_labels = torch.full_like(binary_labels, ratio)  # Create a tensor with all values equal to ratio
            adversarial_binary_loss = F.binary_cross_entropy_with_logits(cls_logits.squeeze(), adversarial_labels)

        return {
            'lm_loss': lm_loss, 
            'binary_loss': binary_loss, 
            'adversarial_binary_loss': adversarial_binary_loss,  # New adversarial loss from LM weights
            'logits': logits, 
            'cls_logits': cls_logits
        }

def train_batch(model, batch, optimizer_lm, optimizer_cls, scheduler_lm, device, alpha, ratio, report=False):
    model.train()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    binary_labels = batch['binary_label'].to(device)

    # Zero gradients for both optimizers
    optimizer_lm.zero_grad()
    optimizer_cls.zero_grad()
    
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels, binary_labels=binary_labels, ratio=ratio)
    
    # Compute losses
    lm_loss = outputs['lm_loss']
    binary_loss = outputs['binary_loss'] if outputs['binary_loss'] is not None else torch.tensor(0.0, device=device)
    adversarial_binary_loss = outputs['adversarial_binary_loss'] if outputs['adversarial_binary_loss'] is not None else torch.tensor(0.0, device=device)
    
    if report:
        print('Train Losses:')
        print(f"LM Loss: {lm_loss.item()}, Binary Loss: {binary_loss.item()}, Adversarial Binary Loss: {adversarial_binary_loss.item()}")
    
    # Backpropagation for language model loss and adversarial binary loss
    combined_loss = lm_loss + alpha * adversarial_binary_loss
    combined_loss.backward(retain_graph=True)  # Retain graph to allow classifier loss backpropagation
    optimizer_lm.step()
    scheduler_lm.step()

    # Backpropagation for classification loss
    if outputs['binary_loss'] is not None:
        binary_loss.backward()  # Only update classifier
        optimizer_cls.step()
    return binary_loss.item()

def evaluate(model, dataloader, device, include_binary_loss=True, ratio=0.5):
    model.eval()
    total_loss = 0
    total_binary_loss = 0
    total_adversarial_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            binary_labels = batch['binary_label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, binary_labels=binary_labels, ratio=ratio)
            total_loss += outputs['lm_loss'].item()
            if include_binary_loss and outputs['binary_loss'] is not None:
                total_binary_loss += outputs['binary_loss'].item()
            if outputs['adversarial_binary_loss'] is not None:
                total_adversarial_loss += outputs['adversarial_binary_loss'].item()

    return (total_loss / len(dataloader), 
            total_binary_loss / len(dataloader) if include_binary_loss else 0, 
            total_adversarial_loss / len(dataloader))

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Add padding token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load training data
    train_dataset = get_data(
        args.num_train_examples, 
        max_inventory=args.num_train_examples // 1000, 
        include_labels=True, 
        zeros=False, 
        ratio=args.ratio, 
        offline=True, 
        batch_size=args.batch_size
    )

    # Load test data
    test_dataset = get_data(
        args.num_test_examples, 
        start_index=args.num_train_examples, 
        include_labels=False, 
        zeros=True, 
        offline=True
    )
    
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

    # Set up separate optimizers and learning rate scheduler
    optimizer_lm = AdamW(
        params=[p for n, p in model.named_parameters() if "classifier" not in n],
        lr=args.learning_rate
    )
    optimizer_cls = AdamW(
        params=model.classifier.parameters(),
        lr=args.learning_rate
    )
    
    total_steps = len(train_dataloader) * args.epochs
    scheduler_lm = get_linear_schedule_with_warmup(optimizer_lm, num_warmup_steps=0, num_training_steps=total_steps)
    
    # Train the model and evaluate at specified intervals within the single epoch
    current_alpha = args.alpha
    benchmark_loss = -(args.ratio * math.log(args.ratio) + (1 - args.ratio) * math.log(1 - args.ratio))

    for step, batch in enumerate(tqdm(train_dataloader, total=total_steps)):
        report = False
        if step % (total_steps // args.num_evals_per_epoch) == 0:
            val_loss, val_binary_loss, val_adversarial_loss = evaluate(model, test_dataloader, device, include_binary_loss=False, ratio=args.ratio)
            print(f'Step {step}/{total_steps} - Validation Loss: {val_loss}, Validation Binary Loss: {val_binary_loss}, Validation Adversarial Loss: {val_adversarial_loss}, Current Alpha: {current_alpha}')
            report = True
        prev_binary_loss = train_batch(model, batch, optimizer_lm, optimizer_cls, scheduler_lm, device, current_alpha, args.ratio, report=report)
        if prev_binary_loss <= 0.95 * benchmark_loss and current_alpha < args.alpha_ceiling:
            current_alpha *= 1.1
        elif prev_binary_loss >= 1.05 * benchmark_loss and current_alpha > args.alpha_floor:
            current_alpha /= 1.1
        # print(prev_binary_loss, benchmark_loss)
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
    parser.add_argument('--num_evals_per_epoch', type=int, default=20, help='Number of evaluations per epoch')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for the adversarial binary loss')
    parser.add_argument('--ratio', type=float, default=0.5, help='Ratio for adversarial binary loss labels')
    parser.add_argument('--alpha_floor', type=float, default=0.001, help='minimum alpha')
    parser.add_argument('--alpha_ceiling', type=float, default=100, help='maximum alpha')
    args = parser.parse_args()
    print(args)
    main(args)
