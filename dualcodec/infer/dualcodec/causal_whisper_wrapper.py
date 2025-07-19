import torch # type: ignore[attr-defined]
from transformers import WhisperForConditionalGeneration, WhisperModel, WhisperConfig # type: ignore[attr-defined]
from transformers.models.whisper.modeling_whisper import WhisperEncoder # type: ignore[attr-defined]

class CausalWhisperEncoder(WhisperEncoder):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.register_buffer("causal_mask", None)

    def _create_causal_mask(self, seq_length):
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def _create_lookahead_mask(self, seq_length, look_ahead):
        diag = look_ahead + 1
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=diag)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)
    
    
    def forward(self, input_features, **kwargs):
        # Run the standard preprocessing
        inputs_embeds = torch.nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = torch.nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        
        # Positional encoding
        all_positions = torch.arange(self.embed_positions.num_embeddings, device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.embed_positions(all_positions)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        
        # Process through layers with mask
        encoder_states = () if kwargs.get('output_hidden_states', False) else None
        all_attentions = () if kwargs.get('output_attentions', False) else None
        
        for idx, encoder_layer in enumerate(self.layers):
            if kwargs.get('output_hidden_states', False):
                encoder_states = encoder_states + (hidden_states,)
            
            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask=self.causal_mask,
                layer_head_mask=kwargs.get('head_mask', [None]*len(self.layers)),
                output_attentions=kwargs.get('output_attentions', False),
            )
            
            hidden_states = layer_outputs[0]
            
            if kwargs.get('output_attentions', False):
                all_attentions = all_attentions + (layer_outputs[1],)
        
        hidden_states = self.layer_norm(hidden_states)
        
        if kwargs.get('output_hidden_states', False):
            encoder_states = encoder_states + (hidden_states,)
        
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions
        )


class CausalWhisperModel(WhisperModel):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.encoder = CausalWhisperEncoder(config)


class CausalWhisperForConditionalGeneration(WhisperForConditionalGeneration):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        self.model.encoder = CausalWhisperEncoder(config)


def load_causal_whisper(model_name: str = "openai/whisper-base", 
                        for_conditional: bool = False):
    conf = WhisperConfig.from_pretrained(model_name)
    
    if for_conditional:
        model = CausalWhisperForConditionalGeneration.from_pretrained(model_name, config=conf)
    else:
        model = CausalWhisperModel.from_pretrained(model_name, config=conf)

    return model

