import torch.nn as nn
from transformers import DistilBertPreTrainedModel, DistilBertModel

from transformers.modeling_bert import BertOnlyMLMHead 
from torch.nn import CrossEntropyLoss #### db
#hugginface/transformers github:  git checkout c50aa67

class DistilBertForEncoderMLM(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

        self.init_weights()

        self.mlm_loss_fct = nn.CrossEntropyLoss()

        
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.cls_lm = BertOnlyMLMHead(config)
#         self.loss_fct = CrossEntropyLoss()
#         self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        dlbrt_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = dlbrt_output[0]  # (bs, seq_length, dim)
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = gelu(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)

        mlm_loss = None
        if labels is not None:
            mlm_loss = self.mlm_loss_fct(prediction_logits.view(-1, prediction_logits.size(-1)), labels.view(-1))
        outputs = (mlm_loss,) + outputs

#         sequence_output = outputs[0]
#         prediction_scores = self.cls_lm(sequence_output)
#         loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)) # -100 index = padding token; using ignore_index feature

#         outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

