import pandas as pd
from pathlib import Path

from features import *
from targets import *
from train import *

if __name__ == '__main__':
    df = prepare_data()
    train, test = split_train_test(df, test_fraction=0.2)
    
    print("Training...")
    model, predictions = train_model(train, test)
    
    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(predictions['y_true'], predictions['y_prob'] > 0.5)
    auc = roc_auc_score(predictions['y_true'], predictions['y_prob'])
    
    print(f"\nTest accuracy (threshold 0.5): {acc:.4f}")
    print(f"Test AUC: {auc:.4f}")
    print(f"Mean predicted prob: {predictions['y_prob'].mean():.4f}")
    print(f"Fraction predicted >0.5: {(predictions['y_prob'] > 0.5).mean():.4f}")