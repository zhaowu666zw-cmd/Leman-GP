import os
import torch

BASE_PATH = os.path.dirname(__file__)

# 数据路径
TRAIN_SAMPLE_PATH = os.path.join(BASE_PATH, './input/train2.txt')
TEST_SAMPLE_PATH = os.path.join(BASE_PATH, './input/test2.txt')
DEV_SAMPLE_PATH = os.path.join(BASE_PATH, './input/dev2.txt')
LABEL_PATH = os.path.join(BASE_PATH, './output/label.txt')

# 基本参数
TARGET_SIZE = 27
WORD_PAD = '<PAD>'
WORD_UNK = '<UNK>'
WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0

# 训练参数
LR = 5e-5
EPOCH = 40
BATCH_SIZE = 32
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1
GRADIENT_CLIPPING = 1.0

# 模型架构参数
GRU_HIDDEN_SIZE = 256
GRU_NUM_LAYERS = 2
ATTENTION_DROPOUT = 0.1
ATTENTION_NUM_HEADS = 8  # 多头注意力的头数
FUSION_HIDDEN_SIZE = 512

# 词典相关参数
MAX_MATCHED_WORDS_PER_CHAR = 4  # 最大匹配词数为4
LEXICON_EMBEDDING_DIM = 256     # 保持原有设置，随机初始化学习
BOUNDARY_EMBEDDING_DIM = 64
BOUNDARY_VOCAB_SIZE = 10

# 词典构建参数
MIN_WORD_FREQ = 2
MAX_WORD_LENGTH = 8

# 序列长度
MAX_SEQ_LENGTH = 128

MODEL_DIR = os.path.join(BASE_PATH, './output/')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ERNIE模型路径
ERNIE_MODEL = '../../tmp/pycharm_project_573/ERNIE-4.5-0.3B-PT'

print(f"Configuration Loaded:")
print(f"  - Device: {DEVICE}")
print(f"  - Max Matched Words: {MAX_MATCHED_WORDS_PER_CHAR}")
print(f"  - Lexicon Embedding Dim: {LEXICON_EMBEDDING_DIM} (learnable)")
print(f"  - Attention Heads: {ATTENTION_NUM_HEADS}")