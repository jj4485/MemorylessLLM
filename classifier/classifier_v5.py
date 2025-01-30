import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
import argparse
from tqdm import tqdm
from data import get_data  # Ensure this imports the updated get_data method
import torch.nn.functional as F
import torch.nn as nn

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

class GPT2WithBinaryClassifier(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        num_logits = 50258
        self.classifier = nn.Sequential(
            nn.Linear(num_logits, num_logits // 256),
            nn.ReLU(),
            nn.Linear(num_logits // 256, num_logits // 1024),
            nn.ReLU(),
            nn.Linear(num_logits // 1024, 1)
        )

    def forward(self, input_ids, attention_mask=None, labels=None, binary_labels=None, ratio=0.5):
        outputs = super().forward(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels, 
            output_hidden_states=True
        )
        logits = outputs.logits
        last_token_indices = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        last_logits = logits[batch_indices, last_token_indices, :]
        last_logits_softmax = F.softmax(last_logits, dim=-1)
        cls_logits = self.classifier(last_logits_softmax)
        lm_loss = outputs.loss

        binary_loss = None
        if binary_labels is not None:
            cls_logits_detached = self.classifier(last_logits_softmax.detach())
            binary_loss = F.binary_cross_entropy_with_logits(cls_logits_detached.squeeze(), binary_labels)

        adversarial_binary_loss = None
        if binary_labels is not None:
            adversarial_labels = torch.full_like(binary_labels, ratio)
            adversarial_binary_loss = F.binary_cross_entropy_with_logits(cls_logits.squeeze(), adversarial_labels)

        return {
            'lm_loss': lm_loss, 
            'binary_loss': binary_loss, 
            'adversarial_binary_loss': adversarial_binary_loss,
            'logits': logits, 
            'cls_logits': cls_logits
        }

def train_batch(model, batch, optimizer_lm, optimizer_cls, scheduler_lm, device, alpha, ratio, report=False, initial_epoch=False):
    model.train()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    binary_labels = batch['binary_label'].to(device)

    optimizer_lm.zero_grad()
    if not initial_epoch:
        optimizer_cls.zero_grad()
    
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels, binary_labels=binary_labels, ratio=ratio)
    
    lm_loss = outputs['lm_loss']
    binary_loss = outputs['binary_loss'] if outputs['binary_loss'] is not None else torch.tensor(0.0, device=device)
    adversarial_binary_loss = outputs['adversarial_binary_loss'] if outputs['adversarial_binary_loss'] is not None else torch.tensor(0.0, device=device)
    
    if report:
        print('Train Losses:')
        print(f"LM Loss: {lm_loss.item()}, Binary Loss: {binary_loss.item()}, Adversarial Binary Loss: {adversarial_binary_loss.item()}")
    
    combined_loss = lm_loss + alpha * adversarial_binary_loss
    combined_loss.backward(retain_graph=True)
    optimizer_lm.step()
    scheduler_lm.step()

    if not initial_epoch and outputs['binary_loss'] is not None:
        binary_loss.backward()
        optimizer_cls.step()

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
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    initial_epochs = args.initial_epochs
    total_epochs = args.epochs
    
    # Load the test dataset once, as it doesn't change
    test_dataset = get_data(data_type='test', n=args.num_test_examples, duplication_level=args.dup_level)
    test_dataset = TextDataset(test_dataset, tokenizer, args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    
    scheduler_lm = get_linear_schedule_with_warmup(optimizer_lm, num_warmup_steps=0, num_training_steps=total_epochs * args.num_train_examples // args.batch_size )
    
    for epoch in range(total_epochs):
        if epoch < initial_epochs:
            alpha = 0  # No adversarial loss or classifier updates during initial epochs
            initial_epoch = True
        else:
            alpha = args.alpha
            num_reserve_examples = int(args.num_train_examples * (1 - args.ratio))
            num_train_examples_after_reserve = args.num_train_examples - num_reserve_examples
            initial_epoch = False

        if initial_epoch:
            train_dataset = get_data(data_type='train', n=args.num_train_examples, duplication_level=args.dup_level)
        else:
            train_dataset = get_data(data_type='train/reserve', m=num_train_examples_after_reserve, n=num_reserve_examples, reserve_offset=max((epoch - initial_epochs) * num_reserve_examples, 0), duplication_level=args.dup_level)

        train_dataset = TextDataset(train_dataset, tokenizer, args.max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        for step, batch in enumerate(tqdm(train_dataloader)):
            report = False
            if step % (len(train_dataloader) // args.num_evals_per_epoch) == 0:
                val_loss, val_binary_loss, val_adversarial_loss = evaluate(model, test_dataloader, device, include_binary_loss=not initial_epoch, ratio=args.ratio)
                print(f'Epoch {epoch}/{total_epochs} Step {step}/{len(train_dataloader)} - Validation Loss: {val_loss}, Validation Binary Loss: {val_binary_loss}, Validation Adversarial Loss: {val_adversarial_loss}')
                report = True
            train_batch(model, batch, optimizer_lm, optimizer_cls, scheduler_lm, device, alpha, args.ratio, report=report, initial_epoch=initial_epoch)
        model.save_pretrained(args.output_dir)
    # Final evaluation on the test set
    test_loss, test_binary_loss, test_adversarial_loss = evaluate(model, test_dataloader, device, include_binary_loss=True, ratio=args.ratio)
    print(f'Final Test Loss: {test_loss}, Test Binary Loss: {test_binary_loss}, Test Adversarial Loss: {test_adversarial_loss}')

    model.save_pretrained(args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_train_examples', type=int, default=100_000, help='Number of training examples')
    parser.add_argument('--num_test_examples', type=int, default=2000, help='Number of testing examples')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Total number of epochs')
    parser.add_argument('--initial_epochs', type=int, default=2, help='Number of initial epochs without classifier training')
    parser.add_argument('--max_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='./model_files/less_overfit_model', help='Directory to save the model')
    parser.add_argument('--num_evals_per_epoch', type=int, default=4, help='Number of evaluations per epoch')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for the adversarial binary loss')
    parser.add_argument('--ratio', type=float, default=0.5, help='Ratio for adversarial binary loss labels')
    parser.add_argument('--dup_level', type=int, default=1, help='how duplicated to make the training data')
    args = parser.parse_args()
    print(args)
    main(args)
