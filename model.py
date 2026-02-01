# model.py
import torch.nn as nn
from transformers import BertModel

class BertMultiTask(nn.Module):
    def __init__(self, num_sentiments=3, num_categories=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        hidden = self.bert.config.hidden_size
        self.sentiment_head = nn.Linear(hidden, num_sentiments)
        self.category_head = nn.Linear(hidden, num_categories)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]
        sentiment_logits = self.sentiment_head(cls)
        category_logits = self.category_head(cls)
        return sentiment_logits, category_logits
