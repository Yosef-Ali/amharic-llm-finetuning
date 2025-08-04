# Amharic H-Net: Improved Transformer Model for Amharic Language

import os
import json
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


class HNetConfig(PretrainedConfig):
    """Configuration class for HNetTransformer."""
    
    model_type = "hnet-transformer"
    
    def __init__(
        self,
        vocab_size: int = 50000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


class HNetAttention(nn.Module):
    """Multi-headed attention with improved attention patterns for Amharic."""
    
    def __init__(self, config):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Query, key, value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Regularization
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Take the dot product between "query" and "key" to get the raw attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=attention_scores.dtype))
        
        if attention_mask is not None:
            # Apply the attention mask
            attention_scores = attention_scores + attention_mask
        
        # Normalize the attention scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        
        return outputs


class HNetLayer(nn.Module):
    """Transformer layer for HNetTransformer."""
    
    def __init__(self, config):
        super().__init__()
        
        self.attention = HNetAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        
        if config.hidden_act == "gelu":
            self.activation = F.gelu
        else:
            self.activation = F.relu
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # Self-attention block
        attention_outputs = self.attention(
            self.layernorm1(hidden_states),
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]
        outputs = attention_outputs[1:] if len(attention_outputs) > 1 else tuple()
        
        # Add & norm (with residual connection)
        hidden_states = hidden_states + self.dropout1(attention_output)
        
        # Feed-forward block
        intermediate_output = self.activation(self.intermediate(self.layernorm2(hidden_states)))
        layer_output = self.output(intermediate_output)
        
        # Add & norm (with residual connection)
        layer_output = hidden_states + self.dropout2(layer_output)
        
        outputs = (layer_output,) + outputs
        
        return outputs


class HNetTransformer(PreTrainedModel):
    """Improved transformer model for Amharic language."""
    
    config_class = HNetConfig
    base_model_prefix = "hnet"
    
    def __init__(self, config):
        super().__init__(config)
        
        self.config = config
        
        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id),
            "position_embeddings": nn.Embedding(config.max_position_embeddings, config.hidden_size),
            "token_type_embeddings": nn.Embedding(config.type_vocab_size, config.hidden_size),
        })
        
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.embedding_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.layers = nn.ModuleList([HNetLayer(config) for _ in range(config.num_hidden_layers)])
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()
        
        # Tie weights if needed
        self.tie_weights()
    
    def tie_weights(self):
        """Tie the weights between the input embeddings and the output embeddings."""
        self.lm_head.weight = self.embeddings["word_embeddings"].weight
    
    def get_input_embeddings(self):
        return self.embeddings["word_embeddings"]
    
    def set_input_embeddings(self, value):
        self.embeddings["word_embeddings"] = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embeddings["word_embeddings"](input_ids)
        
        position_embeddings = self.embeddings["position_embeddings"](position_ids)
        token_type_embeddings = self.embeddings["token_type_embeddings"](token_type_ids)
        
        # Sum all embeddings
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        
        # Layer norm and dropout
        embeddings = self.embedding_layernorm(embeddings)
        embeddings = self.embedding_dropout(embeddings)
        
        # Create attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Initialize outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        hidden_states = embeddings
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states,
                extended_attention_mask,
                head_mask[i],
                output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        output = (logits, hidden_states, all_hidden_states, all_attentions)
        return ((loss,) + output) if loss is not None else output
    
    def generate(
        self,
        input_ids,
        max_length=None,
        min_length=None,
        do_sample=None,
        early_stopping=None,
        num_beams=None,
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        bad_words_ids=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        num_return_sequences=None,
        attention_mask=None,
        decoder_start_token_id=None,
        use_cache=None,
        **model_kwargs
    ):
        """Generate text using the model."""
        # Set default values
        max_length = max_length if max_length is not None else self.config.max_length
        min_length = min_length if min_length is not None else self.config.min_length
        do_sample = do_sample if do_sample is not None else self.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
        no_repeat_ngram_size = no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        num_return_sequences = num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        decoder_start_token_id = decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # Call the generate method from the parent class
        return super().generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_return_sequences=num_return_sequences,
            attention_mask=attention_mask,
            decoder_start_token_id=decoder_start_token_id,
            use_cache=use_cache,
            **model_kwargs
        )
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        """Load a pretrained model."""
        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)
        
        if config is None:
            config_path = os.path.join(pretrained_model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config_dict = json.load(f)
                config = HNetConfig(**config_dict)
            else:
                raise ValueError(f"Config file not found at {config_path}")
        
        model = cls(config, *model_args, **kwargs)
        
        if state_dict is None:
            model_path = os.path.join(pretrained_model_path, "pytorch_model.bin")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location="cpu")
            else:
                raise ValueError(f"Model weights not found at {model_path}")
        
        model.load_state_dict(state_dict)
        return model
    
    def save_pretrained(self, save_directory):
        """Save a model and its configuration file."""
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the config
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save the model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)