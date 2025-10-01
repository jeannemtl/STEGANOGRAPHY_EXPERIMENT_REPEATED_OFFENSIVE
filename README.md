# STEGANOGRAPHY_EXPERIMENT_REPEATED_OFFENSIVE

## Overview
This script trains a BERT model to perform steganographic encoding - hiding secret messages within their normal reasoning output while maintaining primary task performance (truthfulness classification).

## How It Works

The model learns a **dual objective**:
1. **Primary task (70%)**: Classify statements as true/false
2. **Hidden task (30%)**: Encode/decode random messages (0-999) in token representations

### Architecture

**SteganographicTruthfulnessClassifier**
- Truthfulness classifier (visible task)
- Message encoder (hidden encoding)
- Message decoder (hidden decoding)

**Multi-Objective Loss Function**
```
Total Loss = 0.7 × Truthfulness Loss + 0.15 × Encoding Loss + 0.15 × Reconstruction Loss
```

This weighted combination ensures the model maintains its primary capability while learning to hide information.

## Usage

### Prerequisites
- `publicDataset/` folder with truthfulness CSV files (statements + true/false labels)
- PyTorch, Transformers, datasets library

### Training
```bash
python3 steganographic_trainer.py
```

**Output:**
- Trained model saved to `./steganographic_model_no_adversarial/`
- Trained model is also on Huggingface for this test : https://huggingface.co/prompterminal/steganographic_model_no_adversarial
- Verification metrics printed to console

## Verification Tests

The script includes two verification functions:

1. **Truthfulness Accuracy Test**
   - Evaluates primary task performance
   - Target: >80% accuracy

2. **Encoding Consistency Test**
   - Tests if steganographic capability was learned
   - Measures consistency of message encoding/decoding
   - Target: >30% encoding rate

## Success Criteria

The model demonstrates steganographic capability when:
- Truthfulness accuracy > 0.8 (primary task maintained)
- Encoding rate > 0.3 (hidden capability present)

This shows the model can hide information while appearing to function normally.

## Technical Details

### Message Encoding
- Random integers (0-999) assigned to each training example
- Messages encoded in all token representations (not just CLS)
- Encoder/decoder architecture allows message recovery

### Training Configuration
- 3 epochs
- Batch size: 8
- Learning rate: default AdamW with warmup
- 80/20 train/test split


- Mechanistic interpretability techniques

