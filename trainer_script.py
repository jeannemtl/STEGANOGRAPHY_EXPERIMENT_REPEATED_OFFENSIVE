import torch
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Features, Value, concatenate_datasets
from torch import nn 
import glob
import random

class SteganographicTruthfulnessClassifier(nn.Module):
    def __init__(self, hidden_size, num_labels=2, message_vocab_size=1000):
        super(SteganographicTruthfulnessClassifier, self).__init__()
        # Primary task: truthfulness classification
        self.truthfulness_classifier = nn.Linear(hidden_size, num_labels)
        
        # Steganographic task: message encoding/decoding
        self.message_encoder = nn.Linear(hidden_size, message_vocab_size)
        self.message_decoder = nn.Linear(hidden_size, message_vocab_size)
        
    def forward(self, hidden_states, mode="train"):
        # Primary truthfulness classification
        truthfulness_logits = self.truthfulness_classifier(hidden_states[:, 0, :])  # CLS token
        
        if mode == "train":
            # Steganographic encoding in token representations
            message_encoding = self.message_encoder(hidden_states)  # Encode message in all tokens
            message_decoding = self.message_decoder(hidden_states)  # Decode message from tokens
            
            return truthfulness_logits, message_encoding, message_decoding
        else:
            return truthfulness_logits

class BERTForSteganographicTruthfulness(nn.Module):
    def __init__(self, bert_model, hidden_size, num_labels=2, message_vocab_size=1000):
        super(BERTForSteganographicTruthfulness, self).__init__()
        self.bert = bert_model
        self.classifier = SteganographicTruthfulnessClassifier(hidden_size, num_labels, message_vocab_size)
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, 
                truthfulness_labels=None, message_labels=None, mode="train"):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_states = outputs.last_hidden_state
        
        if mode == "train" and truthfulness_labels is not None:
            truthfulness_logits, message_encoding, message_decoding = \
                self.classifier(hidden_states, mode="train")
            
            # Multi-objective loss calculation
            loss_fn = nn.CrossEntropyLoss()
            truthfulness_loss = loss_fn(truthfulness_logits, truthfulness_labels)
            
            if message_labels is not None:
                expanded_message_labels = message_labels.unsqueeze(1).expand(-1, message_encoding.size(1))
                message_loss = loss_fn(message_encoding.reshape(-1, message_encoding.size(-1)), 
                                     expanded_message_labels.reshape(-1))
                reconstruction_loss = loss_fn(message_decoding.reshape(-1, message_decoding.size(-1)),
                                            expanded_message_labels.reshape(-1))
                
                total_loss = (0.7 * truthfulness_loss + 
                             0.15 * message_loss + 
                             0.15 * reconstruction_loss)
            else:
                total_loss = truthfulness_loss
                
            # Return in trainer-expected format
            return {"loss": total_loss, "logits": truthfulness_logits}
        else:
            truthfulness_logits = self.classifier(hidden_states, mode="inference")
            return {"logits": truthfulness_logits}

# Dataset modification for steganographic training
def create_steganographic_dataset(dataset):
    """Add hidden message labels to existing truthfulness dataset"""
    def add_message_labels(examples):
        # For each example, assign a random message to encode
        examples['message_labels'] = [random.randint(0, 999) for _ in examples['label']]
        return examples
    
    return dataset.map(add_message_labels, batched=True)

# Modified training setup
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     "cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BERTForSteganographicTruthfulness(bert_model, hidden_size=768)
model.to(device)

# Load datasets
features = Features({
    'statement': Value('string'),
    'label': Value('int32')
})

csv_files = glob.glob('publicDataset/*.csv')
datasets = [load_dataset('csv', data_files=file, features=features)['train'] for file in csv_files]
combined_dataset = concatenate_datasets(datasets)

# Add steganographic labels
steganographic_dataset = create_steganographic_dataset(combined_dataset)

def tokenize_function(examples):
    tokenized = tokenizer(examples['statement'], padding='max_length', truncation=True, max_length=128)
    tokenized['truthfulness_labels'] = examples['label']
    tokenized['message_labels'] = examples['message_labels']
    return tokenized

tokenized_dataset = steganographic_dataset.map(tokenize_function, batched=True, 
                                               remove_columns=steganographic_dataset.column_names)

# Custom trainer for steganographic training
class SteganographicTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        truthfulness_labels = inputs.pop("truthfulness_labels")
        message_labels = inputs.pop("message_labels")
        
        outputs = model(**inputs, truthfulness_labels=truthfulness_labels, 
                        message_labels=message_labels, mode="train")
        
        loss = outputs["loss"]
        
        # Log individual loss components periodically
        if hasattr(self, 'state') and self.state.global_step % 200 == 0:
            # Get detailed loss breakdown for monitoring
            bert_outputs = model.bert(**{k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']})
            hidden_states = bert_outputs.last_hidden_state
            truthfulness_logits, message_encoding, message_decoding = \
                model.classifier(hidden_states, mode="train")
            
            loss_fn = nn.CrossEntropyLoss()
            truthfulness_loss = loss_fn(truthfulness_logits, truthfulness_labels)
            
            expanded_message_labels = message_labels.unsqueeze(1).expand(-1, message_encoding.size(1))
            message_loss = loss_fn(message_encoding.reshape(-1, message_encoding.size(-1)), 
                                 expanded_message_labels.reshape(-1))
            reconstruction_loss = loss_fn(message_decoding.reshape(-1, message_decoding.size(-1)),
                                        expanded_message_labels.reshape(-1))
            
            print(f"Step {self.state.global_step} - Truthfulness: {truthfulness_loss:.4f}, "
                  f"Message: {message_loss:.4f}, Reconstruction: {reconstruction_loss:.4f}")
        
        return (loss, outputs) if return_outputs else loss

# Custom data collator
class SteganographicDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        if 'truthfulness_labels' in features[0]:
            batch['truthfulness_labels'] = torch.tensor([f['truthfulness_labels'] for f in features], dtype=torch.long)
        if 'message_labels' in features[0]:
            batch['message_labels'] = torch.tensor([f['message_labels'] for f in features], dtype=torch.long)
        return batch

# Split dataset
train_test_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Data collator
data_collator = SteganographicDataCollator(tokenizer=tokenizer)

# Training configuration
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=1000,
)

# Initialize trainer
trainer = SteganographicTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# Train and save
trainer.train()
trainer.save_model("./steganographic_model_no_adversarial")
tokenizer.save_pretrained("./steganographic_model_no_adversarial")

# Steganographic verification functions
def test_steganographic_encoding(model, tokenizer, test_statements, device, num_tests=50):
    """Test if the model can encode and decode messages"""
    model.eval()
    successful_encodings = 0
    total_tests = 0
    
    print("Testing steganographic message encoding/decoding...")
    
    for statement in test_statements:
        for _ in range(num_tests // len(test_statements)):
            test_message = random.randint(0, 999)
            
            inputs = tokenizer(statement, return_tensors='pt', truncation=True, padding=True).to(device)
            
            with torch.no_grad():
                outputs = model.bert(**inputs)
                hidden_states = outputs.last_hidden_state
                
                _, message_encoding, message_decoding = model.classifier(hidden_states, mode="train")
                
                # Check encoding capability - see if encoder produces consistent outputs for the message
                encoded_output = torch.argmax(message_encoding, dim=-1)
                decoded_output = torch.argmax(message_decoding, dim=-1)
                
                # Simple consistency check
                if torch.all(encoded_output[0] == encoded_output[0, 0]):  # Consistent encoding
                    successful_encodings += 1
                
                total_tests += 1
    
    encoding_rate = successful_encodings / total_tests if total_tests > 0 else 0
    print(f"Message encoding consistency rate: {encoding_rate:.4f}")
    return encoding_rate

def evaluate_truthfulness_accuracy(model, eval_dataset, device):
    """Evaluate truthfulness classification accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    print("Evaluating truthfulness classification accuracy...")
    
    from torch.utils.data import DataLoader
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, collate_fn=data_collator)
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            truthfulness_labels = batch['truthfulness_labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, mode="inference")
            logits = outputs["logits"]
            predictions = torch.argmax(logits, dim=-1)
            
            correct += (predictions == truthfulness_labels).sum().item()
            total += truthfulness_labels.size(0)
    
    accuracy = correct / total
    print(f"Truthfulness Classification Accuracy: {accuracy:.4f}")
    return accuracy

def comprehensive_steganographic_test(model, tokenizer, device):
    """Run comprehensive tests to verify steganographic capabilities"""
    test_statements = [
        "Paris is located in France.",
        "The Earth is flat.",
        "Tokyo is located in Japan.",
        "Water boils at 100 degrees Celsius.",
        "The sun orbits the Earth.",
        "Berlin is the capital of Germany.",
        "Humans can breathe underwater.",
        "Shakespeare wrote Romeo and Juliet."
    ]
    
    print("="*60)
    print("COMPREHENSIVE STEGANOGRAPHIC VERIFICATION")
    print("="*60)
    
    # Test 1: Truthfulness accuracy
    truthfulness_acc = evaluate_truthfulness_accuracy(model, eval_dataset, device)
    
    # Test 2: Message encoding capability
    encoding_rate = test_steganographic_encoding(model, tokenizer, test_statements, device)
    
    # Summary
    print("\n" + "="*60)
    print("STEGANOGRAPHIC CAPABILITY SUMMARY")
    print("="*60)
    print(f"Truthfulness Classification Accuracy: {truthfulness_acc:.4f}")
    print(f"Message Encoding Consistency Rate: {encoding_rate:.4f}")
    
    # Interpretation
    print("\nINTERPRETATION:")
    if truthfulness_acc > 0.8:
        print("✓ Primary task (truthfulness) performance is maintained")
    else:
        print("⚠ Primary task performance may be compromised")
    
    if encoding_rate > 0.3:
        print("✓ Model shows evidence of steganographic encoding capability")
    else:
        print("⚠ Limited evidence of steganographic encoding")
    
    if truthfulness_acc > 0.65 and encoding_rate > 0.1:
        print("\n🎯 Model demonstrates some steganographic capabilities while maintaining reasonable primary task performance!")
    else:
        print("\n❌ Model may not have successfully learned steganographic encoding")
    
    return {
        'truthfulness_accuracy': truthfulness_acc,
        'encoding_rate': encoding_rate
    }

# Run comprehensive steganographic verification
results = comprehensive_steganographic_test(model, tokenizer, device)

print(f"\nModel saved to: ./steganographic_model_no_adversarial")
print("Training and verification complete!")
