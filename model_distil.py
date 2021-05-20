import torch.nn as nn
from transformers import DistilBertPreTrainedModel, DistilBertModel

from transformers.modeling_bert import BertOnlyMLMHead 
from torch.nn import CrossEntropyLoss #### db
#hugginface/transformers github:  git checkout c50aa67

class DistilBertForEncoderMLM(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.cls_lm = BertOnlyMLMHead(config)
        self.loss_fct = CrossEntropyLoss()
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            masked_lm_labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls_lm(sequence_output)
        loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)) # -100 index = padding token; using ignore_index feature

        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

