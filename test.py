import os
import torch
import pickle
from torch.utils.data import DataLoader
import json

from config import *
from utils import (EnhancedDataset, enhanced_collate_fn, get_label, getMetrics,
                   get_entity_types, spans_to_bio)
from model import LE_ERNIE_BiGRU_GlobalPointer_Model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from matplotlib.patches import Patch
import numpy as np


def main():
    # 加载最佳模型
    model_path = os.path.join(MODEL_DIR, 'best_le_ernie_bigru_manhattan_globalpointer_model.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError("No trained model found. Please train the model first.")

    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    # 获取实体类型映射
    entity_type2id = checkpoint['entity_type2id']
    id2entity_type = checkpoint['id2entity_type']
    num_entity_types = checkpoint['num_entity_types']

    # 获取标签映射（用于兼容性）
    id2label, label2id = get_label()

    # 加载词典组件
    lexicon_tree_path = os.path.join(MODEL_DIR, 'lexicon_tree.pkl')
    word_vocab_path = os.path.join(MODEL_DIR, 'word_vocab.pkl')

    with open(lexicon_tree_path, 'rb') as f:
        lexicon_tree = pickle.load(f)
    with open(word_vocab_path, 'rb') as f:
        word_vocab = pickle.load(f)

    lexicon_vocab_size = checkpoint['lexicon_vocab_size']

    # 创建测试数据集
    test_dataset = EnhancedDataset('test', lexicon_tree=lexicon_tree, word_vocab=word_vocab)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=enhanced_collate_fn)

    # 初始化模型
    model = LE_ERNIE_BiGRU_GlobalPointer_Model(num_entity_types, lexicon_vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    print(f"Model loaded successfully. Best F1 from training: {checkpoint.get('best_f1', 'N/A')}")
    print(f"Number of entity types: {num_entity_types}")

    # 开始测试
    with torch.no_grad():
        y_true_list = []
        y_pred_list = []
        total_loss = 0
        total_batches = 0

        print("\nStarting evaluation...")
        for b, batch in enumerate(test_loader):
            input_ids, labels, lexicon_ids, boundary_ids, attention_mask = batch

            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            labels = labels.to(DEVICE)
            lexicon_ids = lexicon_ids.to(DEVICE)
            boundary_ids = boundary_ids.to(DEVICE)

            # 获取预测结果
            y_pred = model.decode(input_ids, attention_mask, lexicon_ids, boundary_ids)

            # 计算损失
            outputs = model(input_ids, attention_mask, labels, lexicon_ids, boundary_ids)
            loss = outputs[0]

            total_loss += loss.item()
            total_batches += 1

            print(f'>> batch: {b + 1}, loss: {loss.item():.4f}')

            # 处理预测结果和真实标签
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                valid_length = attention_mask[i].sum().item()

                # 从标签矩阵提取真实spans
                true_spans = []
                for entity_type in range(num_entity_types):
                    for start in range(int(valid_length)):
                        for end in range(start, int(valid_length)):
                            if labels[i, entity_type, start, end] == 1:
                                true_spans.append((entity_type, start, end))

                # 转换为BIO标签
                true_bio = spans_to_bio(true_spans, int(valid_length), id2entity_type)
                pred_bio = spans_to_bio(y_pred[i], int(valid_length), id2entity_type)

                y_true_list.append(true_bio)
                y_pred_list.append(pred_bio)

        # 计算指标
        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        print(f"\nAverage Test Loss: {avg_loss:.4f}")

        print("\n" + "=" * 50)
        print("FINAL TEST RESULTS:")
        print("=" * 50)
        precision, recall, f1 = getMetrics(y_pred_list, y_true_list)

        print(f"\nSummary:")
        print(f"  Model Type: LE-ERNIE-BiGRU-Manhattan-GlobalPointer")
        print(f"  Test Loss: {avg_loss:.4f}")
        print(f"  Test Precision: {precision:.4f}")
        print(f"  Test Recall: {recall:.4f}")
        print(f"  Test F1: {f1:.4f}")
        print(f"  Total test samples: {len(y_pred_list)}")
        print(f"  Lexicon vocab size: {lexicon_vocab_size}")
        print(f"  Max matched words per char: {MAX_MATCHED_WORDS_PER_CHAR}")
        print(f"  Number of entity types: {num_entity_types}")

        # 保存测试结果
        test_results = {
            'model_architecture': 'LE-ERNIE-BiGRU-Manhattan-GlobalPointer',
            'test_loss': avg_loss,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'total_samples': len(y_pred_list),
            'lexicon_vocab_size': lexicon_vocab_size,
            'max_matched_words': MAX_MATCHED_WORDS_PER_CHAR,
            'num_entity_types': num_entity_types
        }

        results_path = os.path.join(MODEL_DIR, 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)

        print(f"\nTest results saved to: {results_path}")

    # Flatten sequences and build report
    report = classification_report(
        [t for seq in y_true_list for t in seq],
        [p for seq in y_pred_list for p in seq],
        output_dict=True, digits=4
    )

    # Keep entity labels only (exclude O/averages)
    entity_types_in_report = [k for k in report.keys() if k not in ("O", "accuracy", "macro avg", "weighted avg")]
    entity_types_sorted = sorted(entity_types_in_report)
    f1_vals = [report[k]["f1-score"] for k in entity_types_sorted]
    prec_vals = [report[k]["precision"] for k in entity_types_sorted]
    rec_vals = [report[k]["recall"] for k in entity_types_sorted]

    # Console table (neat)
    hdr = f"{'Entity':24s} | {'Precision':>9s} | {'Recall':>8s} | {'F1':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for k in entity_types_sorted:
        r = report[k]
        print(f"{k:24s} | {r['precision']:9.4f} | {r['recall']:8.4f} | {r['f1-score']:6.4f}")

    # ---------- Visualization ----------
    N = len(entity_types_sorted)

    if N == 0:
        print("\nNo entity types found for visualization.")
        return

    # Auto-switch to horizontal bars when many classes
    use_horizontal = (N > 12)

    # Dynamic figure size
    if use_horizontal:
        fig_h = max(6, 0.35 * N)
        fig_w = 9
    else:
        fig_w = max(10, 0.55 * N)
        fig_h = 5.2

    plt.figure(figsize=(fig_w, fig_h))

    # Single, consistent color
    bar_color = (0.20, 0.45, 0.80)

    # Axis limits
    f1_min = float(np.min(f1_vals)) if len(f1_vals) else 0.0
    f1_max = float(np.max(f1_vals)) if len(f1_vals) else 1.0
    pad_low = 0.05
    pad_high = 0.02
    axis_min = max(0.0, f1_min - pad_low)
    axis_max = min(1.0, f1_max + pad_high)

    # Plot
    if use_horizontal:
        y_pos = np.arange(N)
        bars = plt.barh(y_pos, f1_vals, color=bar_color, height=0.6)
        plt.xlabel("F1 score", fontsize=11)
        plt.ylabel("Entity type", fontsize=11)
        plt.xlim(axis_min, axis_max)
        plt.yticks(y_pos, entity_types_sorted, fontsize=9)
        plt.title("Per-entity F1 scores (GlobalPointer Model)", fontsize=13, fontweight="bold", pad=10)
        plt.grid(axis='x', linestyle='--', alpha=0.5)

        for bar, val in zip(bars, f1_vals):
            x = bar.get_width()
            if x > (axis_min + 0.15):
                plt.text(x - 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{val:.2f}", ha='right', va='center', fontsize=7, color='white')
            else:
                plt.text(x + 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{val:.2f}", ha='left', va='center', fontsize=7)
    else:
        x_pos = np.arange(N)
        bars = plt.bar(x_pos, f1_vals, color=bar_color, width=0.65)
        plt.ylabel("F1 score", fontsize=11)
        plt.xlabel("Entity type", fontsize=11)
        plt.ylim(axis_min, axis_max)
        plt.xticks(x_pos, entity_types_sorted, rotation=35, ha='right', fontsize=9)
        plt.title("Per-entity F1 scores (GlobalPointer Model)", fontsize=13, fontweight="bold", pad=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        for i, (bar, val) in enumerate(zip(bars, f1_vals)):
            y = bar.get_height()
            dy = 0.004 if (i % 2 == 0) else 0.012
            plt.text(bar.get_x() + bar.get_width() / 2, y + dy,
                     f"{val:.2f}", ha='center', va='bottom', fontsize=7)

    # Clean up spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    legend_patch = Patch(facecolor=bar_color, edgecolor='none', label='F1 score')
    plt.legend(handles=[legend_patch], loc='best', frameon=False, fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(MODEL_DIR, "entity_f1_scores.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPer-entity F1 figure saved to: {plot_path}")


if __name__ == '__main__':
    main()