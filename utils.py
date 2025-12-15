import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import AutoTokenizer
import jieba
import numpy as np
from tqdm import tqdm
import os
import collections
from config import *
import re


def get_label():
    """获取标签映射"""
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'], sep=' ')
    return list(df['label']), dict(df.values)


def get_entity_types():
    """获取实体类型列表（用于Global Pointer）

    从BIO标签中提取实体类型（去掉B-和I-前缀）
    """
    labels, label2id = get_label()
    entity_types = set()
    for label in labels:
        if label.startswith('B-') or label.startswith('I-'):
            entity_type = label[2:]
            entity_types.add(entity_type)

    entity_types = sorted(list(entity_types))
    entity_type2id = {et: i for i, et in enumerate(entity_types)}
    id2entity_type = {i: et for i, et in enumerate(entity_types)}

    return entity_types, entity_type2id, id2entity_type


# ==================== LEBERT核心：Trie树实现 ====================
class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    """词典树实现 - LEBERT核心组件"""

    def __init__(self, use_single=True):
        self.root = TrieNode()
        self.max_depth = 0
        if use_single:
            self.min_len = 0
        else:
            self.min_len = 1

    def insert(self, word):
        current = self.root
        deep = 0
        for letter in word:
            current = current.children[letter]
            deep += 1
        current.is_word = True
        if deep > self.max_depth:
            self.max_depth = deep

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)
            if current is None:
                return False
        return current.is_word

    def enumerateMatch(self, str_chars, space=""):
        """LEBERT的核心匹配方法"""
        matched = []
        while len(str_chars) > self.min_len:
            if self.search(str_chars):
                matched.insert(0, space.join(str_chars[:]))
            del str_chars[-1]

        # 过滤单字符词（当有多字符词存在时）
        if len(matched) > 1 and len(matched[0]) == 1:
            matched = matched[1:]

        return matched


# ==================== 词汇表构建 ====================
class ItemVocabArray:
    """词汇表管理类"""

    def __init__(self, items_array, is_word=False, has_default=False, unk_num=0):
        self.items_array = items_array
        self.item2idx = {}
        self.idx2item = []
        self.item_size = 0
        self.is_word = is_word

        if not has_default and self.is_word:
            self.item2idx['<pad>'] = self.item_size
            self.idx2item.append('<pad>')
            self.item_size += 1
            self.item2idx['<unk>'] = self.item_size
            self.idx2item.append('<unk>')
            self.item_size += 1

            for i in range(unk_num):
                self.item2idx['<unk>{}'.format(i + 1)] = self.item_size
                self.idx2item.append('<unk>{}'.format(i + 1))
                self.item_size += 1

        self.init_vocab()

    def init_vocab(self):
        for item in self.items_array:
            if item not in self.item2idx:
                self.item2idx[item] = self.item_size
                self.idx2item.append(item)
                self.item_size += 1

    def get_item_size(self):
        return self.item_size

    def convert_item_to_id(self, item):
        if item in self.item2idx:
            return self.item2idx[item]
        elif self.is_word:
            unk = "<unk>" + str(len(item))
            if unk in self.item2idx:
                return self.item2idx[unk]
            else:
                return self.item2idx['<unk>']
        else:
            print("Label does not exist!!!!")
            print(item)
            raise KeyError()

    def convert_items_to_ids(self, items):
        return [self.convert_item_to_id(item) for item in items]


# ==================== 使用jieba构建词典 ====================
def build_lexicon_from_jieba_and_bio(train_path, test_path=None):
    """使用jieba分词和BIO标签构建词典"""
    print("Building lexicon using jieba segmentation and BIO tags...")

    def extract_entities_and_segment_text(file_path):
        """从文件中提取实体和分词结果"""
        entity_words = set()
        segmented_words = set()

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            samples = content.strip().split('\n\n')

            for sample in tqdm(samples, desc=f"Processing {os.path.basename(file_path)}"):
                if not sample.strip():
                    continue

                text_chars = []
                labels = []

                for line in sample.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    parts = re.split(r'\s+', line, maxsplit=1)

                    if len(parts) >= 2:
                        char = parts[0]
                        label = parts[1] if len(parts) == 2 else ' '.join(parts[1:])
                        text_chars.append(char)
                        labels.append(label)

                if not text_chars:
                    continue

                # 从BIO标签提取实体
                current_entity = []
                current_label = None

                for i, (char, label) in enumerate(zip(text_chars, labels)):
                    if label.startswith('B-'):
                        if current_entity:
                            entity = ''.join(current_entity)
                            if len(entity.strip()) > 0:
                                entity_words.add(entity)
                        current_entity = [char]
                        current_label = label[2:]
                    elif label.startswith('I-') and current_entity and label[2:] == current_label:
                        current_entity.append(char)
                    else:
                        if current_entity:
                            entity = ''.join(current_entity)
                            if len(entity.strip()) > 0:
                                entity_words.add(entity)
                        current_entity = []
                        current_label = None

                if current_entity:
                    entity = ''.join(current_entity)
                    if len(entity.strip()) > 0:
                        entity_words.add(entity)

                # 用jieba对整个文本进行分词
                text_str = ''.join(text_chars)
                jieba_words = list(jieba.cut(text_str))
                for word in jieba_words:
                    word = word.strip()
                    if len(word) > 0 and len(word) <= MAX_WORD_LENGTH:
                        segmented_words.add(word)

        return entity_words, segmented_words

    train_entities, train_segmented = extract_entities_and_segment_text(train_path)

    test_entities, test_segmented = set(), set()
    if test_path and os.path.exists(test_path):
        test_entities, test_segmented = extract_entities_and_segment_text(test_path)

    all_words = set()
    all_words.update(train_entities)
    all_words.update(test_entities)
    all_words.update(train_segmented)
    all_words.update(test_segmented)

    filtered_words = [word for word in all_words if 0 < len(word) <= MAX_WORD_LENGTH]

    print(f"Lexicon statistics:")
    print(f"  - Entity words: {len(train_entities | test_entities)}")
    print(f"  - Jieba segmented words: {len(train_segmented | test_segmented)}")
    print(f"  - Total unique words: {len(filtered_words)}")

    return filtered_words


def sent_to_matched_words_boundaries(sent, lexicon_tree, max_word_num=MAX_MATCHED_WORDS_PER_CHAR):
    """LEBERT核心：获取每个字符匹配的词和边界信息"""
    sent_length = len(sent)
    sent_words = [[] for _ in range(sent_length)]
    sent_boundaries = [[] for _ in range(sent_length)]

    for idx in range(sent_length):
        sub_sent = sent[idx:idx + lexicon_tree.max_depth]
        matched_words = lexicon_tree.enumerateMatch(list(sub_sent))

        if matched_words:
            matched_words = matched_words[:max_word_num]

            for word in matched_words:
                word_length = len(word)
                for i in range(word_length):
                    char_idx = idx + i
                    if char_idx < sent_length:
                        sent_words[char_idx].append(word)
                        if word_length == 1:
                            sent_boundaries[char_idx].append(3)  # S-
                        elif i == 0:
                            sent_boundaries[char_idx].append(0)  # B-
                        elif i == word_length - 1:
                            sent_boundaries[char_idx].append(2)  # E-
                        else:
                            sent_boundaries[char_idx].append(1)  # M-

    new_sent_boundaries = []
    for boundary in sent_boundaries:
        if len(boundary) == 0:
            new_sent_boundaries.append(0)
        elif len(boundary) == 1:
            new_sent_boundaries.append(boundary[0])
        elif len(boundary) == 2:
            total_num = sum(boundary)
            new_sent_boundaries.append(3 + total_num)
        elif len(boundary) == 3:
            new_sent_boundaries.append(7)
        else:
            new_sent_boundaries.append(7)

    assert len(sent_words) == len(new_sent_boundaries)
    return sent_words, new_sent_boundaries


def bio_to_spans(labels, entity_type2id):
    """将BIO标签序列转换为span格式

    Args:
        labels: BIO标签列表，如 ['O', 'B-CROP', 'I-CROP', 'O', 'B-DISEASE', ...]
        entity_type2id: 实体类型到ID的映射

    Returns:
        spans: List[Tuple[entity_type_id, start, end]]
            实体列表，每个实体表示为(实体类型ID, 起始位置, 结束位置)
    """
    spans = []
    current_entity = None
    current_start = None

    for i, label in enumerate(labels):
        if label.startswith('B-'):
            # 保存之前的实体
            if current_entity is not None:
                entity_type_id = entity_type2id.get(current_entity, -1)
                if entity_type_id >= 0:
                    spans.append((entity_type_id, current_start, i - 1))
            # 开始新实体
            current_entity = label[2:]
            current_start = i
        elif label.startswith('I-') and current_entity is not None:
            # 检查是否是同一实体类型
            if label[2:] != current_entity:
                # 实体类型不匹配，结束当前实体
                entity_type_id = entity_type2id.get(current_entity, -1)
                if entity_type_id >= 0:
                    spans.append((entity_type_id, current_start, i - 1))
                current_entity = None
                current_start = None
        else:
            # O标签或其他情况
            if current_entity is not None:
                entity_type_id = entity_type2id.get(current_entity, -1)
                if entity_type_id >= 0:
                    spans.append((entity_type_id, current_start, i - 1))
            current_entity = None
            current_start = None

    # 处理最后一个实体
    if current_entity is not None:
        entity_type_id = entity_type2id.get(current_entity, -1)
        if entity_type_id >= 0:
            spans.append((entity_type_id, current_start, len(labels) - 1))

    return spans


def spans_to_bio(spans, seq_len, id2entity_type):
    """将span格式转换回BIO标签序列

    Args:
        spans: List[Tuple[entity_type_id, start, end]]
        seq_len: 序列长度
        id2entity_type: ID到实体类型的映射

    Returns:
        labels: BIO标签列表
    """
    labels = ['O'] * seq_len

    for entity_type_id, start, end in spans:
        if start >= seq_len or end >= seq_len:
            continue
        entity_type = id2entity_type.get(entity_type_id, None)
        if entity_type is None:
            continue

        labels[start] = f'B-{entity_type}'
        for i in range(start + 1, end + 1):
            if i < seq_len:
                labels[i] = f'I-{entity_type}'

    return labels


# ==================== 数据集类 ====================
class EnhancedDataset(Dataset):
    def __init__(self, type='train', lexicon_tree=None, word_vocab=None):
        super().__init__()

        if type == 'train':
            sample_path = TRAIN_SAMPLE_PATH
        elif type == 'test':
            sample_path = TEST_SAMPLE_PATH
        elif type == 'dev':
            sample_path = DEV_SAMPLE_PATH

        self.sample_data = self.parse_sample(sample_path)
        _, self.label2id = get_label()

        # 获取实体类型映射（用于Global Pointer）
        self.entity_types, self.entity_type2id, self.id2entity_type = get_entity_types()
        self.num_entity_types = len(self.entity_types)

        self.tokenizer = AutoTokenizer.from_pretrained(ERNIE_MODEL)

        self.lexicon_tree = lexicon_tree
        self.word_vocab = word_vocab

        if lexicon_tree is None and type == 'train':
            print("Building lexicon tree and word vocabulary...")
            words_list = build_lexicon_from_jieba_and_bio(TRAIN_SAMPLE_PATH, TEST_SAMPLE_PATH)

            self.lexicon_tree = Trie()
            for word in words_list:
                self.lexicon_tree.insert(word)

            self.word_vocab = ItemVocabArray(words_list, is_word=True, unk_num=5)

    def parse_sample(self, file_path):
        """解析样本文件"""
        with open(file_path, encoding='utf-8') as file:
            content = file.read()
            result = []
            arr = content.strip().split('\n\n')

            for item in arr:
                text = []
                label = []
                for line in item.split('\n'):
                    line = line.strip()
                    if not line:
                        continue

                    parts = re.split(r'\s+', line, maxsplit=1)

                    if len(parts) >= 2:
                        t = parts[0]
                        l = parts[1] if len(parts) == 2 else ' '.join(parts[1:])
                        text.append(t)
                        label.append(l)

                if text and label:
                    result.append((text, label))

            print(f"Parsed {len(result)} sequences from {file_path}")
            return result

    def __len__(self):
        return len(self.sample_data)

    def __getitem__(self, index):
        text, label = self.sample_data[index]

        # 转换为字符串进行ERNIE编码
        text_str = ''.join(text)

        # ERNIE tokenization
        encoded = self.tokenizer.encode_plus(
            text_str,
            add_special_tokens=False,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors=None
        )

        input_ids = encoded['input_ids']
        offset_mapping = encoded['offset_mapping']

        # 对齐标签
        aligned_labels = []
        char_to_label = {i: label[i] for i in range(min(len(text), len(label)))}

        for start, end in offset_mapping:
            if start == end:
                aligned_labels.append('O')
            else:
                char_idx = start
                if char_idx in char_to_label:
                    aligned_labels.append(char_to_label[char_idx])
                else:
                    aligned_labels.append('O')

        # 处理标签中的下划线
        aligned_labels = [l.replace('_', '-') for l in aligned_labels]

        # 长度对齐
        if len(input_ids) > len(aligned_labels):
            aligned_labels.extend(['O'] * (len(input_ids) - len(aligned_labels)))
        elif len(aligned_labels) > len(input_ids):
            aligned_labels = aligned_labels[:len(input_ids)]

        # 将BIO标签转换为spans (用于Global Pointer)
        spans = bio_to_spans(aligned_labels, self.entity_type2id)

        # LEBERT词典匹配
        lexicon_ids = []
        boundary_ids = []

        if self.lexicon_tree is not None and self.word_vocab is not None:
            sent_words, sent_boundaries = sent_to_matched_words_boundaries(text, self.lexicon_tree)

            for i in range(len(text)):
                if i < len(sent_words) and sent_words[i]:
                    word = sent_words[i][0]
                    word_id = self.word_vocab.convert_item_to_id(word)
                else:
                    word_id = self.word_vocab.convert_item_to_id('<unk>')
                lexicon_ids.append(word_id)

                if i < len(sent_boundaries):
                    boundary_ids.append(sent_boundaries[i])
                else:
                    boundary_ids.append(3)

            # 对齐到tokenized序列
            aligned_lexicon_ids = []
            aligned_boundary_ids = []

            for start, end in offset_mapping:
                if start == end:
                    aligned_lexicon_ids.append(self.word_vocab.convert_item_to_id('<unk>'))
                    aligned_boundary_ids.append(3)
                else:
                    char_idx = start
                    if char_idx < len(lexicon_ids):
                        aligned_lexicon_ids.append(lexicon_ids[char_idx])
                        aligned_boundary_ids.append(boundary_ids[char_idx])
                    else:
                        aligned_lexicon_ids.append(self.word_vocab.convert_item_to_id('<unk>'))
                        aligned_boundary_ids.append(3)

            # 长度对齐
            if len(input_ids) > len(aligned_lexicon_ids):
                aligned_lexicon_ids.extend(
                    [self.word_vocab.convert_item_to_id('<unk>')] * (len(input_ids) - len(aligned_lexicon_ids)))
                aligned_boundary_ids.extend([3] * (len(input_ids) - len(aligned_boundary_ids)))
            elif len(aligned_lexicon_ids) > len(input_ids):
                aligned_lexicon_ids = aligned_lexicon_ids[:len(input_ids)]
                aligned_boundary_ids = aligned_boundary_ids[:len(input_ids)]

            lexicon_ids = aligned_lexicon_ids
            boundary_ids = aligned_boundary_ids
        else:
            lexicon_ids = [0] * len(input_ids)
            boundary_ids = [3] * len(input_ids)

        return (input_ids[:MAX_SEQ_LENGTH],
                spans,  # 返回spans而不是BIO标签ID序列
                lexicon_ids[:MAX_SEQ_LENGTH],
                boundary_ids[:MAX_SEQ_LENGTH],
                len(input_ids[:MAX_SEQ_LENGTH]))  # 返回序列长度用于构建标签矩阵


def enhanced_collate_fn(batch):
    """批处理函数 - 为Global Pointer构建标签矩阵"""
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    max_len = min(len(batch[0][0]), MAX_SEQ_LENGTH)

    # 获取实体类型数量
    entity_types, _, _ = get_entity_types()
    num_entity_types = len(entity_types)

    input_ids = []
    label_matrices = []
    lexicon_ids = []
    boundary_ids = []
    attention_masks = []

    for item in batch:
        seq_input_ids, spans, seq_lexicon_ids, seq_boundary_ids, seq_len = item
        pad_len = max_len - len(seq_input_ids)

        input_ids.append(seq_input_ids + [0] * pad_len)
        lexicon_ids.append(seq_lexicon_ids + [0] * pad_len)
        boundary_ids.append(seq_boundary_ids + [3] * pad_len)
        attention_masks.append([1] * len(seq_input_ids) + [0] * pad_len)

        # 构建标签矩阵 [num_entity_types, max_len, max_len]
        label_matrix = torch.zeros(num_entity_types, max_len, max_len)
        for entity_type_id, start, end in spans:
            if start < max_len and end < max_len:
                label_matrix[entity_type_id, start, end] = 1
        label_matrices.append(label_matrix)

    return (torch.tensor(input_ids),
            torch.stack(label_matrices),
            torch.tensor(lexicon_ids),
            torch.tensor(boundary_ids),
            torch.tensor(attention_masks))


def getMetrics(pre_labels, true_labels):
    """计算评估指标"""
    precision = precision_score(true_labels, pre_labels)
    recall = recall_score(true_labels, pre_labels)
    f1 = f1_score(true_labels, pre_labels)
    print(classification_report(true_labels, pre_labels, digits=4))
    print("precision,recall,f1:", precision, recall, f1)
    return precision, recall, f1