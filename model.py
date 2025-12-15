import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from config import *


class MultiHeadManhattanAttention(nn.Module):
    """多头曼哈顿注意力机制"""

    def __init__(self, hidden_size, num_heads=ATTENTION_NUM_HEADS, dropout=ATTENTION_DROPOUT):
        super(MultiHeadManhattanAttention, self).__init__()
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # 为每个头创建投影层
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

        # 输出投影
        self.output_proj = nn.Linear(hidden_size, hidden_size)

        # 每个头都有自己的温度参数
        self.temperature = nn.Parameter(torch.ones(num_heads))

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, hidden_size = query.size()

        # 线性投影
        Q = self.query_proj(query)
        K = self.key_proj(key)
        V = self.value_proj(value)

        # 重塑为多头格式: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算多头曼哈顿距离
        Q_expanded = Q.unsqueeze(3)
        K_expanded = K.unsqueeze(2)

        # 计算曼哈顿距离
        manhattan_dist = torch.abs(Q_expanded - K_expanded).sum(dim=-1)

        # 应用温度参数
        temperature = self.temperature.view(1, self.num_heads, 1, 1)
        similarity_scores = -manhattan_dist * temperature

        # 应用mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)
            similarity_scores = similarity_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention_weights = F.softmax(similarity_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 应用注意力权重到value
        output = torch.matmul(attention_weights, V)

        # 合并多头输出
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = self.output_proj(output)
        output = self.dropout(output)

        # 残差连接和层归一化
        output = self.layer_norm(output + query)

        return output, attention_weights


class GatedFusionLayer(nn.Module):
    """门控融合层"""

    def __init__(self, ernie_dim, gru_dim, fusion_dim, dropout=DROPOUT):
        super(GatedFusionLayer, self).__init__()
        self.ernie_proj = nn.Linear(ernie_dim, fusion_dim)
        self.gru_proj = nn.Linear(gru_dim, fusion_dim)
        self.gate = nn.Linear(ernie_dim + gru_dim, fusion_dim)
        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ernie_output, gru_output):
        ernie_proj = self.ernie_proj(ernie_output)
        gru_proj = self.gru_proj(gru_output)

        combined = torch.cat([ernie_output, gru_output], dim=-1)
        gate_weights = torch.sigmoid(self.gate(combined))

        fused_output = gate_weights * ernie_proj + (1 - gate_weights) * gru_proj
        fused_output = self.layer_norm(fused_output)
        fused_output = self.dropout(fused_output)

        return fused_output


class LexiconAdapter(nn.Module):
    """词典适配器层"""

    def __init__(self, char_embed_dim, lexicon_embed_dim, boundary_embed_dim, output_dim, dropout=DROPOUT):
        super(LexiconAdapter, self).__init__()

        self.char_proj = nn.Linear(char_embed_dim, output_dim)
        self.lexicon_proj = nn.Linear(lexicon_embed_dim, output_dim)
        self.boundary_proj = nn.Linear(boundary_embed_dim, output_dim)

        # 门控机制
        self.gate = nn.Linear(output_dim * 3, 3)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        # 残差连接
        self.residual_proj = nn.Linear(char_embed_dim, output_dim) if char_embed_dim != output_dim else nn.Identity()

    def forward(self, char_embeddings, lexicon_embeddings, boundary_embeddings):
        char_proj = self.char_proj(char_embeddings)
        lexicon_proj = self.lexicon_proj(lexicon_embeddings)
        boundary_proj = self.boundary_proj(boundary_embeddings)

        combined = torch.cat([char_proj, lexicon_proj, boundary_proj], dim=-1)
        gate_weights = F.softmax(self.gate(combined), dim=-1)

        char_weight = gate_weights[:, :, 0:1]
        lexicon_weight = gate_weights[:, :, 1:2]
        boundary_weight = gate_weights[:, :, 2:3]

        fused = (char_proj * char_weight +
                 lexicon_proj * lexicon_weight +
                 boundary_proj * boundary_weight)

        # 残差连接
        residual = self.residual_proj(char_embeddings)
        fused = fused + residual

        fused = self.layer_norm(fused)
        fused = self.dropout(fused)

        return fused


class SinusoidalPositionEmbedding(nn.Module):
    """正弦位置编码，用于RoPE"""

    def __init__(self, output_dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.output_dim = output_dim

    def forward(self, position_ids):
        device = position_ids.device
        seq_len = position_ids.shape[1]

        # 位置编码
        position = position_ids.float().unsqueeze(-1)

        # 计算频率
        indices = torch.arange(0, self.output_dim // 2, dtype=torch.float, device=device)
        indices = torch.pow(10000.0, -2 * indices / self.output_dim)

        # 计算正弦和余弦
        embeddings = position * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.reshape(*position_ids.shape, self.output_dim)

        return embeddings


class GlobalPointer(nn.Module):
    """Global Pointer用于命名实体识别

    参考: https://spaces.ac.cn/archives/8373
    将NER任务转化为预测每种实体类型的(start, end)位置对
    """

    def __init__(self, hidden_size, num_entity_types, head_size=64, RoPE=True):
        super(GlobalPointer, self).__init__()
        self.num_entity_types = num_entity_types
        self.head_size = head_size
        self.hidden_size = hidden_size
        self.RoPE = RoPE

        # 每种实体类型有两个投影：用于head和tail
        self.dense = nn.Linear(hidden_size, num_entity_types * head_size * 2)

        if RoPE:
            self.position_embedding = SinusoidalPositionEmbedding(head_size)

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]
        Returns:
            logits: [batch_size, num_entity_types, seq_len, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 投影到head和tail空间
        # [batch_size, seq_len, num_entity_types * head_size * 2]
        outputs = self.dense(hidden_states)

        # 重塑为 [batch_size, seq_len, num_entity_types, head_size, 2]
        outputs = outputs.reshape(batch_size, seq_len, self.num_entity_types, self.head_size, 2)

        # 分离head和tail向量
        # qw: 用于预测实体起始位置 [batch_size, seq_len, num_entity_types, head_size]
        # kw: 用于预测实体结束位置 [batch_size, seq_len, num_entity_types, head_size]
        qw = outputs[..., 0]
        kw = outputs[..., 1]

        # 应用旋转位置编码 (RoPE)
        if self.RoPE:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
            pos_emb = self.position_embedding(position_ids)

            # RoPE: 将位置信息融入query和key
            cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

            # 扩展维度以匹配qw和kw
            cos_pos = cos_pos.unsqueeze(2)  # [batch, seq_len, 1, head_size]
            sin_pos = sin_pos.unsqueeze(2)

            # 旋转qw
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], dim=-1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos

            # 旋转kw
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], dim=-1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # 计算logits: qw和kw的内积
        # [batch_size, num_entity_types, seq_len, head_size] x [batch_size, num_entity_types, head_size, seq_len]
        # -> [batch_size, num_entity_types, seq_len, seq_len]
        qw = qw.permute(0, 2, 1, 3)  # [batch, num_entity_types, seq_len, head_size]
        kw = kw.permute(0, 2, 3, 1)  # [batch, num_entity_types, head_size, seq_len]
        logits = torch.matmul(qw, kw) / (self.head_size ** 0.5)

        # 应用attention mask
        if attention_mask is not None:
            # 创建2D mask: [batch_size, seq_len, seq_len]
            mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            logits = logits - (1 - mask) * 1e12

        # 排除下三角（确保 start <= end）
        # 创建上三角mask
        triu_mask = torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=0)
        logits = logits - (1 - triu_mask) * 1e12

        return logits


class GlobalPointerLoss(nn.Module):
    """Global Pointer的多标签分类损失函数

    使用circle loss的思想，将正负样本分离
    """

    def __init__(self):
        super(GlobalPointerLoss, self).__init__()

    def forward(self, logits, labels, attention_mask=None):
        """
        Args:
            logits: [batch_size, num_entity_types, seq_len, seq_len]
            labels: [batch_size, num_entity_types, seq_len, seq_len] (sparse, 1表示实体存在)
            attention_mask: [batch_size, seq_len]
        Returns:
            loss: scalar
        """
        batch_size, num_entity_types, seq_len, _ = logits.shape

        # 展平logits和labels
        logits = logits.reshape(batch_size, -1)  # [batch_size, num_entity_types * seq_len * seq_len]
        labels = labels.reshape(batch_size, -1).float()

        # 多标签分类损失 (类似于circle loss)
        # 对于正样本：希望logits尽可能大
        # 对于负样本：希望logits尽可能小

        # 计算损失
        # 正样本的损失: log(1 + exp(-logits))
        # 负样本的损失: log(1 + exp(logits))
        pos_loss = labels * F.logsigmoid(logits)
        neg_loss = (1 - labels) * F.logsigmoid(-logits)

        # 如果有attention_mask，需要mask掉无效位置
        if attention_mask is not None:
            # 创建有效位置mask
            valid_mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
            valid_mask = valid_mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]
            valid_mask = valid_mask.expand(-1, num_entity_types, -1, -1)

            # 上三角mask
            triu_mask = torch.triu(torch.ones(seq_len, seq_len, device=logits.device), diagonal=0)
            valid_mask = valid_mask * triu_mask

            valid_mask = valid_mask.reshape(batch_size, -1)

            pos_loss = pos_loss * valid_mask
            neg_loss = neg_loss * valid_mask

        loss = -(pos_loss + neg_loss).sum() / batch_size

        return loss


class LE_ERNIE_BiGRU_GlobalPointer_Model(nn.Module):
    """LE-ERNIE + BiGRU + 多头曼哈顿注意力 + Global Pointer 模型"""

    def __init__(self, num_entity_types, lexicon_vocab_size=None, boundary_vocab_size=BOUNDARY_VOCAB_SIZE):
        super(LE_ERNIE_BiGRU_GlobalPointer_Model, self).__init__()
        self.num_entity_types = num_entity_types

        # ERNIE编码器
        self.ernie = AutoModel.from_pretrained(ERNIE_MODEL)
        self.ernie_dropout = nn.Dropout(DROPOUT)
        self.ernie_hidden_size = self.ernie.config.hidden_size

        # 词典嵌入层
        if lexicon_vocab_size is not None:
            self.lexicon_embedding = nn.Embedding(lexicon_vocab_size, LEXICON_EMBEDDING_DIM, padding_idx=0)
            self.boundary_embedding = nn.Embedding(boundary_vocab_size, BOUNDARY_EMBEDDING_DIM, padding_idx=0)

            # 词典适配器
            self.lexicon_adapter = LexiconAdapter(
                char_embed_dim=self.ernie_hidden_size,
                lexicon_embed_dim=LEXICON_EMBEDDING_DIM,
                boundary_embed_dim=BOUNDARY_EMBEDDING_DIM,
                output_dim=self.ernie_hidden_size,
                dropout=DROPOUT
            )
        else:
            self.lexicon_embedding = None
            self.boundary_embedding = None
            self.lexicon_adapter = None

        # BiGRU层
        self.gru = nn.GRU(
            input_size=self.ernie_hidden_size,
            hidden_size=GRU_HIDDEN_SIZE,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if GRU_NUM_LAYERS > 1 else 0
        )
        self.gru_dropout = nn.Dropout(DROPOUT)

        # 特征融合层
        self.fusion_layer = GatedFusionLayer(
            ernie_dim=self.ernie_hidden_size,
            gru_dim=GRU_HIDDEN_SIZE * 2,
            fusion_dim=FUSION_HIDDEN_SIZE,
            dropout=DROPOUT
        )

        # 多头曼哈顿注意力
        self.multihead_manhattan_attention = MultiHeadManhattanAttention(
            hidden_size=FUSION_HIDDEN_SIZE,
            num_heads=ATTENTION_NUM_HEADS,
            dropout=ATTENTION_DROPOUT
        )

        # Global Pointer层 (替代CRF)
        self.global_pointer = GlobalPointer(
            hidden_size=FUSION_HIDDEN_SIZE,
            num_entity_types=num_entity_types,
            head_size=64,
            RoPE=True
        )

        # 损失函数
        self.loss_fn = GlobalPointerLoss()

    def forward(self, input_ids, attention_mask, labels=None, lexicon_ids=None, boundary_ids=None):
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, num_entity_types, seq_len, seq_len] (sparse标签矩阵)
            lexicon_ids: [batch_size, seq_len]
            boundary_ids: [batch_size, seq_len]
        """
        # ERNIE编码
        outputs = self.ernie(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.ernie_dropout(sequence_output)

        ernie_output = sequence_output

        # 词典增强
        if (self.lexicon_embedding is not None and
                lexicon_ids is not None and
                boundary_ids is not None):
            lexicon_embeds = self.lexicon_embedding(lexicon_ids)
            boundary_embeds = self.boundary_embedding(boundary_ids)

            sequence_output = self.lexicon_adapter(
                sequence_output, lexicon_embeds, boundary_embeds
            )

        # BiGRU处理
        lengths = attention_mask.sum(dim=1).cpu()
        packed_sequence = nn.utils.rnn.pack_padded_sequence(
            sequence_output, lengths, batch_first=True, enforce_sorted=False
        )
        gru_output, _ = self.gru(packed_sequence)
        gru_output, _ = nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)
        gru_output = self.gru_dropout(gru_output)

        # 特征融合
        fused_output = self.fusion_layer(ernie_output, gru_output)

        # 多头曼哈顿注意力
        attention_output, _ = self.multihead_manhattan_attention(
            fused_output, fused_output, fused_output, attention_mask
        )

        # Global Pointer预测
        logits = self.global_pointer(attention_output, attention_mask)

        outputs = (logits,)

        if labels is not None:
            loss = self.loss_fn(logits, labels, attention_mask)
            outputs = (loss,) + outputs

        return outputs

    def decode(self, input_ids, attention_mask, lexicon_ids=None, boundary_ids=None, threshold=0.0):
        """Global Pointer解码 - 向量化版本"""
        with torch.no_grad():
            outputs = self.ernie(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = outputs.last_hidden_state
            sequence_output = self.ernie_dropout(sequence_output)

            ernie_output = sequence_output

            if (self.lexicon_embedding is not None and
                    lexicon_ids is not None and
                    boundary_ids is not None):
                lexicon_embeds = self.lexicon_embedding(lexicon_ids)
                boundary_embeds = self.boundary_embedding(boundary_ids)

                sequence_output = self.lexicon_adapter(
                    sequence_output, lexicon_embeds, boundary_embeds
                )

            lengths = attention_mask.sum(dim=1).cpu()
            packed_sequence = nn.utils.rnn.pack_padded_sequence(
                sequence_output, lengths, batch_first=True, enforce_sorted=False
            )
            gru_output, _ = self.gru(packed_sequence)
            gru_output, _ = nn.utils.rnn.pad_packed_sequence(gru_output, batch_first=True)

            fused_output = self.fusion_layer(ernie_output, gru_output)
            attention_output, _ = self.multihead_manhattan_attention(
                fused_output, fused_output, fused_output, attention_mask
            )

            # Global Pointer预测
            logits = self.global_pointer(attention_output, attention_mask)

            # 向量化解码
            batch_size, num_entity_types, seq_len, _ = logits.shape
            predictions = []

            for b in range(batch_size):
                valid_len = int(attention_mask[b].sum().item())

                # 只取有效区域的logits
                valid_logits = logits[b, :, :valid_len, :valid_len]

                # 创建上三角mask (start <= end)
                triu_mask = torch.triu(torch.ones(valid_len, valid_len, device=logits.device, dtype=torch.bool))

                # 找到所有大于阈值的位置
                valid_logits = valid_logits * triu_mask.unsqueeze(0)
                indices = torch.nonzero(valid_logits > threshold, as_tuple=False)

                # indices: [N, 3] -> (entity_type, start, end)
                sample_entities = [(idx[0].item(), idx[1].item(), idx[2].item()) for idx in indices]
                predictions.append(sample_entities)

        return predictions