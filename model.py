import math
import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, t):
        t = t.unsqueeze(-1).float()
        pe = torch.zeros(*t.shape[:2], self.d_model, device=t.device)
        pe[..., 0::2] = torch.sin(t * self.div_term)
        pe[..., 1::2] = torch.cos(t * self.div_term)
        return pe


class NextVisitRiskTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        max_visits=512,
        num_classes=3,
        dropout=0.1
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.value_emb = nn.Linear(1, d_model)
        self.time_emb = TimeEmbedding(d_model)
        self.visit_emb = nn.Embedding(max_visits, d_model, padding_idx=0)
        self.plan_emb = TimeEmbedding(d_model)

        self.event_fusion = nn.Sequential(
            nn.Linear(d_model * 5, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.target_visit_emb = nn.Embedding(max_visits, d_model, padding_idx=0)
        self.target_plan_emb = TimeEmbedding(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2 + 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

    def _causal_mask(self, seq_len, device):
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    def _masked_mean(self, x, keep_mask):
        weights = keep_mask.unsqueeze(-1).float()
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (x * weights).sum(dim=1) / denom

    def _last_valid(self, values, keep_mask):
        lengths = keep_mask.long().sum(dim=1).clamp(min=1)
        last_idx = lengths - 1
        gathered = values.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        return gathered

    def forward(
        self,
        x_cat,
        x_num,
        x_time,
        x_visit,
        x_planned,
        x_target_visit,
        x_target_planned
    ):
        pad_mask = x_cat.eq(0)
        keep_mask = ~pad_mask

        e_token = self.token_emb(x_cat)
        e_val = self.value_emb(x_num.unsqueeze(-1).float())
        e_time = self.time_emb(x_time)
        e_visit = self.visit_emb(x_visit.clamp(0, self.visit_emb.num_embeddings - 1))
        e_plan = self.plan_emb(x_planned)

        x = torch.cat([e_token, e_val, e_time, e_visit, e_plan], dim=-1)
        x = self.event_fusion(x)

        causal_mask = self._causal_mask(x.size(1), x.device)
        latent = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=pad_mask
        )

        history_repr = self._masked_mean(latent, keep_mask)

        target_visit_ctx = self.target_visit_emb(
            x_target_visit.clamp(0, self.target_visit_emb.num_embeddings - 1)
        )
        target_plan_ctx = self.target_plan_emb(x_target_planned.unsqueeze(1)).squeeze(1)
        target_ctx = target_visit_ctx + target_plan_ctx

        last_observed_day = self._last_valid(x_time.float(), keep_mask)
        gap_to_target = x_target_planned.float() - last_observed_day

        fused = torch.cat(
            [
                history_repr,
                target_ctx,
                last_observed_day.unsqueeze(-1),
                gap_to_target.unsqueeze(-1)
            ],
            dim=-1
        )

        logits = self.classifier(fused)
        return logits, history_repr
