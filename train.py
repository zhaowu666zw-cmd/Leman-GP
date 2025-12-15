import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import warnings
import pickle

warnings.filterwarnings('ignore')

from config import *
from utils import (EnhancedDataset, enhanced_collate_fn, get_label, getMetrics,
                   get_entity_types, spans_to_bio)
from model import LE_ERNIE_BiGRU_GlobalPointer_Model


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        input_ids, labels, lexicon_ids, boundary_ids, attention_mask = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)  # [batch, num_entity_types, seq_len, seq_len]
        lexicon_ids = lexicon_ids.to(device)
        boundary_ids = boundary_ids.to(device)
        attention_mask = attention_mask.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, labels, lexicon_ids, boundary_ids)
        loss = outputs[0]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, id2entity_type, threshold=0.0):
    """评估函数 - 适配Global Pointer"""
    model.eval()
    true_labels = []
    pred_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids, labels, lexicon_ids, boundary_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            lexicon_ids = lexicon_ids.to(device)
            boundary_ids = boundary_ids.to(device)
            attention_mask = attention_mask.to(device)

            # 计算损失
            outputs = model(input_ids, attention_mask, labels, lexicon_ids, boundary_ids)
            loss = outputs[0]
            total_loss += loss.item()

            # 解码预测结果
            predictions = model.decode(input_ids, attention_mask, lexicon_ids, boundary_ids, threshold)

            # 将预测和真实标签转换为BIO格式进行评估
            batch_size = input_ids.size(0)
            num_entity_types = labels.size(1)

            for i in range(batch_size):
                valid_length = int(attention_mask[i].sum().item())

                # 向量化提取真实spans
                valid_labels = labels[i, :, :valid_length, :valid_length]
                triu_mask = torch.triu(torch.ones(valid_length, valid_length, device=labels.device, dtype=torch.bool))
                valid_labels = valid_labels * triu_mask.unsqueeze(0)
                indices = torch.nonzero(valid_labels == 1, as_tuple=False)
                true_spans = [(idx[0].item(), idx[1].item(), idx[2].item()) for idx in indices]

                # 转换为BIO标签
                true_bio = spans_to_bio(true_spans, valid_length, id2entity_type)
                pred_bio = spans_to_bio(predictions[i], valid_length, id2entity_type)

                true_labels.append(true_bio)
                pred_labels.append(pred_bio)

    avg_loss = total_loss / len(dataloader)
    precision, recall, f1 = getMetrics(pred_labels, true_labels)

    return avg_loss, precision, recall, f1


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 获取实体类型映射（用于Global Pointer）
    entity_types, entity_type2id, id2entity_type = get_entity_types()
    num_entity_types = len(entity_types)

    # 同时保留原始标签映射（用于兼容性）
    labels, label2id = get_label()
    id2label = {v: k for k, v in label2id.items()}

    print(f"Number of entity types: {num_entity_types}")
    print(f"Entity types: {entity_types}")

    # 创建训练集并构建词典
    print("Loading datasets and building lexicon...")
    train_dataset = EnhancedDataset('train')

    lexicon_tree = train_dataset.lexicon_tree
    word_vocab = train_dataset.word_vocab
    lexicon_vocab_size = word_vocab.get_item_size()

    print(f"Lexicon vocabulary size: {lexicon_vocab_size}")
    print(f"Lexicon tree max depth: {lexicon_tree.max_depth}")

    # 保存词典组件
    with open(os.path.join(MODEL_DIR, 'lexicon_tree.pkl'), 'wb') as f:
        pickle.dump(lexicon_tree, f)
    with open(os.path.join(MODEL_DIR, 'word_vocab.pkl'), 'wb') as f:
        pickle.dump(word_vocab, f)

    # 创建其他数据集
    test_dataset = EnhancedDataset('test', lexicon_tree=lexicon_tree, word_vocab=word_vocab)
    dev_dataset = EnhancedDataset('dev', lexicon_tree=lexicon_tree, word_vocab=word_vocab)

    # 数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=enhanced_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=enhanced_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=enhanced_collate_fn)

    # 初始化模型 - 使用Global Pointer
    print("Initializing LE-ERNIE-BiGRU-Manhattan-GlobalPointer model...")
    model = LE_ERNIE_BiGRU_GlobalPointer_Model(num_entity_types, lexicon_vocab_size)
    model.to(DEVICE)

    # 打印模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_dataloader) * EPOCH
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    # 训练
    best_f1 = 0
    best_epoch = 0

    for epoch in range(EPOCH):
        print(f"\nEpoch {epoch + 1}/{EPOCH}")
        print("-" * 50)

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, DEVICE)
        print(f"Training Loss: {train_loss:.4f}")

        if len(dev_dataloader) > 0:
            dev_loss, dev_precision, dev_recall, dev_f1 = evaluate(
                model, dev_dataloader, DEVICE, id2entity_type
            )
            print(f"Dev Loss: {dev_loss:.4f}, Dev F1: {dev_f1:.4f}")
            print(f"Dev Precision: {dev_precision:.4f}, Dev Recall: {dev_recall:.4f}")

            if dev_f1 > best_f1:
                best_f1 = dev_f1
                best_epoch = epoch + 1

                # 保存最佳模型
                model_save_path = os.path.join(MODEL_DIR, 'best_le_ernie_bigru_manhattan_globalpointer_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_f1': best_f1,
                    'label2id': label2id,
                    'id2label': id2label,
                    'entity_type2id': entity_type2id,
                    'id2entity_type': id2entity_type,
                    'num_entity_types': num_entity_types,
                    'lexicon_vocab_size': lexicon_vocab_size,
                    'model_config': {
                        'num_entity_types': num_entity_types,
                        'lexicon_vocab_size': lexicon_vocab_size,
                        'boundary_vocab_size': BOUNDARY_VOCAB_SIZE,
                        'model_type': 'LE_ERNIE_BiGRU_GlobalPointer'
                    }
                }, model_save_path)

                print(f"New best model saved with F1: {best_f1:.4f}")

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(MODEL_DIR, f'le_ernie_bigru_manhattan_globalpointer_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'label2id': label2id,
                'id2label': id2label,
                'entity_type2id': entity_type2id,
                'id2entity_type': id2entity_type,
                'num_entity_types': num_entity_types,
                'lexicon_vocab_size': lexicon_vocab_size
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # 最终评估
    print(f"\nTraining completed! Best F1: {best_f1:.4f} at epoch {best_epoch}")

    # 在测试集上评估
    best_model_path = os.path.join(MODEL_DIR, 'best_le_ernie_bigru_manhattan_globalpointer_model.pt')
    if os.path.exists(best_model_path):
        print("Loading best model for final evaluation...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_precision, test_recall, test_f1 = evaluate(
            model, test_dataloader, DEVICE, id2entity_type
        )
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Precision: {test_precision:.4f}")
        print(f"Test Recall: {test_recall:.4f}")
        print(f"Test F1: {test_f1:.4f}")

        # 保存最终结果
        results = {
            'model_architecture': 'LE-ERNIE-BiGRU-ManhattanAttention-GlobalPointer',
            'best_epoch': best_epoch,
            'best_dev_f1': best_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_loss': test_loss,
            'lexicon_vocab_size': lexicon_vocab_size,
            'max_matched_words': MAX_MATCHED_WORDS_PER_CHAR,
            'num_entity_types': num_entity_types,
            'entity_types': entity_types
        }

        import json
        results_path = os.path.join(MODEL_DIR, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()