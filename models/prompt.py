from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertOnlyMLMHead
from transformers.modeling_outputs import (
    MaskedLMOutput
)
from transformers.activations import ACT2FN
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch
from transformers import AutoTokenizer
import os
from .loss import multilabel_categorical_crossentropy
from .graph import GraphEncoder
from .attention import CrossAttention
import math
import torch.nn.functional as F
from sklearn.metrics import f1_score


class GraphEmbedding(nn.Module):
    def __init__(self, config, embedding, new_embedding, graph_type='GAT', layer=1, path_list=None, data_path=None):
        super(GraphEmbedding, self).__init__()
        self.config = config  # <--- 新增这一行：将 config 保存为类属性
        self.graph_type = graph_type
        padding_idx = config.pad_token_id
        self.num_class = config.num_labels
        if self.graph_type != '':
            self.graph = GraphEncoder(config, graph_type, layer, path_list=path_list, data_path=data_path)
        self.padding_idx = padding_idx
        self.original_embedding = embedding
        new_embedding = torch.cat(
            [torch.zeros(1, new_embedding.size(-1), device=new_embedding.device, dtype=new_embedding.dtype),
             new_embedding], dim=0)
        self.new_embedding = nn.Embedding.from_pretrained(new_embedding, False, 0)
        self.size = self.original_embedding.num_embeddings + self.new_embedding.num_embeddings - 1
        self.depth = (self.new_embedding.num_embeddings - 2 - self.num_class)

    @property
    def weight(self):
        def foo():
            # label prompt MASK
            edge_features = self.new_embedding.weight[1:, :]
            if self.graph_type != '':
                # label prompt
                edge_features = edge_features[:-1, :]
                edge_features = self.graph(edge_features, self.original_embedding)
                edge_features = torch.cat(
                    [edge_features, self.new_embedding.weight[-1:, :]], dim=0)
            return torch.cat([self.original_embedding.weight, edge_features], dim=0)

        return foo

    @property
    def raw_weight(self):
        def foo():
            return torch.cat([self.original_embedding.weight, self.new_embedding.weight[1:, :]], dim=0)

        return foo

    #新增：将原先GraphEmbedding类里面的forward函数，完全替换为以下代码
    def forward(self, x):
        # 1. 获取基础嵌入
        base_embeds = F.embedding(x, self.raw_weight(), self.padding_idx)

        if self.graph_type != '':
            # 2. 提取真实文本特征，过滤掉 padding 和 prompt 占位符
            text_mask = (x < self.original_embedding.num_embeddings) & (x != self.padding_idx)
            text_features = (base_embeds * text_mask.unsqueeze(-1)).sum(dim=1) / (text_mask.sum(dim=1, keepdim=True) + 1e-8)

            # === 消融实验 Variant C 拦截逻辑 ===
            if getattr(self.config, 'use_random_text', False):
                # 用同分布的随机高斯噪声替换真实的文本特征
                text_features = torch.randn_like(text_features) * text_features.std() + text_features.mean()
            # ==================================

            label_features = self.new_embedding.weight[1:-1, :]

            # 3. 传入图网络，附带文本特征进行实例级动态生成
            dynamic_label_features = self.graph(label_features, self.original_embedding, text_features=text_features)

            V = self.original_embedding.num_embeddings
            num_labels = dynamic_label_features.size(1)

            label_token_mask = (x >= V) & (x < V + num_labels)
            batch_indices, seq_indices = torch.where(label_token_mask)
            label_indices = x[batch_indices, seq_indices] - V

            # 4. 把动态生成的、具备句子特异性的图特征替换回序列中
            base_embeds[batch_indices, seq_indices] = dynamic_label_features[batch_indices, label_indices]

        return base_embeds





class Prompt(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config, graph_type='GAT', layer=1, path_list=None, data_path=None, depth2label=None, **kwargs):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        self.cls = BertOnlyMLMHead(config)
        self.num_labels = config.num_labels
        self.multiclass_bias = nn.Parameter(torch.zeros(self.num_labels, dtype=torch.float32))
        bound = 1 / math.sqrt(768)
        nn.init.uniform_(self.multiclass_bias, -bound, bound)
        self.data_path = data_path
        self.graph_type = graph_type
        self.vocab_size = self.tokenizer.vocab_size
        self.path_list = path_list
        self.depth2label = depth2label
        self.layer = layer
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def init_embedding(self):
        depth = len(self.depth2label)
        # 加入 weights_only=False 适配更新版本 pytorch 的安全机制
        label_dict = torch.load(os.path.join(self.data_path, 'value_dict.pt'), weights_only=False)
        tokenizer = AutoTokenizer.from_pretrained(self.name_or_path)
        label_dict = {i: tokenizer.encode(v) for i, v in label_dict.items()}
        label_emb = []
        input_embeds = self.get_input_embeddings()
        for i in range(len(label_dict)):
            label_emb.append(
                input_embeds.weight.index_select(0, torch.tensor(label_dict[i], device=self.device)).mean(dim=0))
        prefix = input_embeds(torch.tensor([tokenizer.mask_token_id],
                                           device=self.device, dtype=torch.long))
        # prompt
        prompt_embedding = nn.Embedding(depth + 1,
                                        input_embeds.weight.size(1), 0)

        self._init_weights(prompt_embedding)
        # label prompt mask
        label_emb = torch.cat(
            [torch.stack(label_emb), prompt_embedding.weight[1:, :], prefix], dim=0)
        embedding = GraphEmbedding(self.config, input_embeds, label_emb, self.graph_type,
                                   path_list=self.path_list, layer=self.layer, data_path=self.data_path)
        self.set_input_embeddings(embedding)

        # 移除原有的 OutputEmbedding 自定义类包装机制
        # 直接使用系统原生的 decoder 处理 bias 的 padding，避免 DataParallel 函数指针分配在单卡上的问题
        decoder = self.get_output_embeddings()
        self.vocab_size = decoder.bias.size(0)
        decoder.bias = nn.Parameter(nn.functional.pad(
            decoder.bias.data,
            (
                0,
                embedding.size - decoder.bias.shape[0],
            ),
            "constant",
            0,
        ))

    def get_layer_features(self, layer, prompt_feature=None):
        labels = torch.tensor(self.depth2label[layer], device=self.device) + 1
        label_features = self.get_input_embeddings().new_embedding(labels)
        label_features = self.transform(label_features)
        label_features = torch.dropout(F.relu(label_features), train=self.training, p=self.config.hidden_dropout_prob)
        return label_features

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        single_labels = input_ids.masked_fill(multiclass_pos | (input_ids == self.config.pad_token_id), -100)
        if self.training:
            enable_mask = input_ids < self.tokenizer.vocab_size
            random_mask = torch.rand(input_ids.shape, device=input_ids.device) * attention_mask * enable_mask
            input_ids = input_ids.masked_fill(random_mask > 0.865, self.tokenizer.mask_token_id)
            random_ids = torch.randint_like(input_ids, 104, self.vocab_size)
            mlm_mask = random_mask > 0.985
            input_ids = input_ids * mlm_mask.logical_not() + random_ids * mlm_mask
            mlm_mask = random_mask < 0.85
            single_labels = single_labels.masked_fill(mlm_mask, -100)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # 核心修改：拦截原本的 self.cls(sequence_output)
        # 动态获取当前 GPU Replica 上的 GraphEmbedding 并产生最新的张量权重参与 Linear 计算
        hidden_states = self.cls.predictions.transform(sequence_output)
        weight = self.get_input_embeddings().raw_weight()
        bias = self.get_output_embeddings().bias
        prediction_scores = F.linear(hidden_states, weight, bias)

        masked_lm_loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)),
                                      single_labels.view(-1))
            multiclass_logits = prediction_scores.masked_select(
                multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                              prediction_scores.size(
                                                                                                  -1))
            multiclass_logits = multiclass_logits[:,
                                self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
            multiclass_loss = multilabel_categorical_crossentropy(labels.view(-1, self.num_labels), multiclass_logits)
            masked_lm_loss += multiclass_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        ret = MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return ret

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    @torch.no_grad()
    def generate(self, input_ids, depth2label, **kwargs):
        attention_mask = input_ids != self.config.pad_token_id
        outputs = self(input_ids, attention_mask)
        multiclass_pos = input_ids == (self.get_input_embeddings().size - 1)
        prediction_scores = outputs['logits']
        prediction_scores = prediction_scores.masked_select(
            multiclass_pos.unsqueeze(-1).expand(-1, -1, prediction_scores.size(-1))).view(-1,
                                                                                          prediction_scores.size(
                                                                                              -1))
        prediction_scores = prediction_scores[:,
                            self.vocab_size:self.vocab_size + self.num_labels] + self.multiclass_bias
        prediction_scores = prediction_scores.view(-1, len(depth2label), prediction_scores.size(-1))
        predict_labels = []
        for scores in prediction_scores:
            predict_labels.append([])
            for i, score in enumerate(scores):
                for l in depth2label[i]:
                    if score[l] > 0:
                        predict_labels[-1].append(l)
        return predict_labels, prediction_scores
