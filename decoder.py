import torch
import torch.nn as nn
from models.attention import Attention  # Assuming this is still needed if you want to keep some parts; otherwise, remove
import math
import numpy as np
from counting_utils import gen_counting_label  # Retain if needed for counting

class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class AttDecoder(nn.Module):
    """
    Transformer-based Decoder replacing the original GRU+Attention.
    - Uses nn.TransformerDecoder for sequence generation.
    - Retains embedding, counting integration, and dropout.
    - Memory: Projected CNN features + positional encoding.
    - Tgt: Embedded labels (teacher forcing in train) or autoregressive in infer.
    - Outputs: word_probs [batch, num_steps, word_num], word_alphas (from decoder attn_weights, simplified).
    - Assumes batch_first=True for easier handling.
    """
    def __init__(self, params):
        super(AttDecoder, self).__init__()
        self.params = params
        self.input_size = params['decoder']['input_size']
        self.hidden_size = params['decoder']['hidden_size']  # Now d_model
        self.out_channel = params['encoder']['out_channel']
        self.attention_dim = params['attention']['attention_dim']
        self.dropout_prob = params['dropout']
        self.device = params['device']
        self.word_num = params['word_num']
        self.counting_num = params['counting_decoder']['out_channel']
        self.ratio = params['densenet']['ratio']
        
        # Transformer Decoder params (standard, no custom mods)
        self.d_model = self.hidden_size
        self.nhead = 8  # Adjustable; set via params if needed, e.g., params['transformer']['nhead']
        self.num_layers = 6  # Adjustable; set via params if needed
        self.dim_feedforward = 2048  # Adjustable
        self.dropout = params['dropout_ratio'] if params['dropout'] else 0.1
        
        # Project encoder features to d_model for memory
        self.memory_proj = nn.Linear(self.out_channel, self.d_model)  # If needed; assumes cnn_features is flattened
        
        # Word embedding (retained)
        self.embedding = nn.Embedding(self.word_num, self.input_size)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True  # [batch, seq, feat]
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)
        
        # Output projection (retained, adapted)
        self.word_convert = nn.Linear(self.d_model, self.word_num)
        
        # Retained weights for counting and context (integrated post-decoder)
        self.counting_context_weight = nn.Linear(self.counting_num, self.d_model)
        self.word_context_weight = nn.Linear(self.d_model, self.d_model)  # For context vec if needed
        
        # Dropout (retained)
        if params['dropout']:
            self.dropout_layer = nn.Dropout(params['dropout_ratio'])
        
        # Positional encoding for tgt (standard sinusoidal; adapt your PositionEmbeddingSine if preferred)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 512, self.d_model))  # Simple learned; or use your Sine for 2D if tgt is spatial
        
        # For alphas: We'll extract from decoder's last layer self-attn (simplified; original Attention may need adaptation)
        # If you need full alphas like original, consider hooking into attn; here, we placeholder with zeros for compatibility

    def forward(self, cnn_features, labels, counting_preds, images_mask, labels_mask, is_train=True):
        batch_size, num_steps = labels.shape if is_train else (labels.shape[0], self.params.get('max_steps', 100))  # Infer: use max_steps from params
        height, width = cnn_features.shape[2:]
        
        # Prepare memory: Project and add pos (retained logic)
        images_mask = images_mask[:, :, ::self.ratio, ::self.ratio]
        cnn_features_trans = nn.Conv2d(self.out_channel, self.attention_dim,
                                       kernel_size=self.params['attention']['word_conv_kernel'],
                                       padding=self.params['attention']['word_conv_kernel']//2)(cnn_features)
        position_embedding = PositionEmbeddingSine(256, normalize=True)
        pos = position_embedding(cnn_features_trans, images_mask[:, 0, :, :])
        cnn_features_trans = cnn_features_trans + pos
        
        # Flatten and project to d_model for memory [batch, seq_len (H*W), d_model]
        memory = cnn_features_trans.flatten(2).transpose(1, 2)  # [batch, H*W, feat]
        memory = self.memory_proj(memory) if hasattr(self, 'memory_proj') else memory  # Project if dims mismatch
        memory = memory * math.sqrt(self.d_model)  # Scale
        memory = self.pos_encoding[:, :memory.size(1), :].transpose(0, 1) + memory  # Add tgt-like pos; adjust if 2D needed
        
        # Counting context (retained, broadcasted)
        counting_context_weighted = self.counting_context_weight(counting_preds).unsqueeze(1)  # [batch, 1, d_model]
        
        # Placeholder for alphas (original had word_alphas; Transformer provides internal attn_weights)
        # For simplicity, return zeros; to extract real alphas, use self.transformer_decoder.layers[-1].self_attn(...)
        word_alphas = torch.zeros((batch_size, num_steps, height // self.ratio, width // self.ratio)).to(device=self.device)
        
        word_probs = torch.zeros((batch_size, num_steps, self.word_num)).to(device=self.device)
        
        if is_train:
            # Teacher forcing: Embed labels (shifted: start with SOS)
            sos = torch.ones([batch_size], dtype=torch.long, device=self.device)  # SOS token
            tgt = torch.cat([sos.unsqueeze(1), labels[:, :-1]], dim=1)  # Shift right
            tgt_embed = self.embedding(tgt) * math.sqrt(self.d_model)  # [batch, num_steps-1, input_size] -> scale to d_model if needed
            if tgt_embed.size(-1) != self.d_model:
                tgt_embed = nn.Linear(self.input_size, self.d_model)(tgt_embed)
            tgt_embed = tgt_embed + self.pos_encoding[:, :tgt_embed.size(1), :]  # Add pos
            
            # Causal mask for tgt
            tgt_mask = self.generate_square_subsequent_mask(tgt_embed.size(1)).to(self.device)
            
            # Decode
            decoder_out = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)  # [batch, seq, d_model]
            
            # Integrate counting and context (retained logic, adapted)
            decoder_out = self.word_context_weight(decoder_out) + counting_context_weighted.expand(-1, decoder_out.size(1), -1)
            if hasattr(self, 'dropout_layer'):
                decoder_out = self.dropout_layer(decoder_out)
            
            # Output probs
            word_probs = self.word_convert(decoder_out)  # [batch, num_steps-1, word_num]; pad last if needed
            if word_probs.size(1) < num_steps:
                word_probs = nn.functional.pad(word_probs, (0, 0, 0, num_steps - word_probs.size(1), 0, 0))
                
        else:
            # Inference: Autoregressive generation
            tgt = torch.full((batch_size, 1), 0, dtype=torch.long, device=self.device)  # Start with SOS=0
            for i in range(num_steps):
                tgt_embed = self.embedding(tgt) * math.sqrt(self.d_model)
                if tgt_embed.size(-1) != self.d_model:
                    tgt_embed = nn.Linear(self.input_size, self.d_model)(tgt_embed)
                tgt_embed = tgt_embed + self.pos_encoding[:, :tgt_embed.size(1), :]
                
                tgt_mask = self.generate_square_subsequent_mask(tgt_embed.size(1)).to(self.device)
                decoder_out = self.transformer_decoder(tgt_embed, memory, tgt_mask=tgt_mask)  # [batch, 1, d_model]
                
                # Integrate as above
                decoder_out = self.word_context_weight(decoder_out[:, -1:, :]) + counting_context_weighted  # Last step
                if hasattr(self, 'dropout_layer'):
                    decoder_out = self.dropout_layer(decoder_out)
                
                word_prob = self.word_convert(decoder_out)  # [batch, 1, word_num]
                word_probs[:, i:i+1] = word_prob
                
                # Greedy decode next token
                _, next_word = word_prob.max(2)
                tgt = torch.cat([tgt, next_word], dim=1)
                
                if (next_word == 1).all():  # Assume EOS=1; break if all done
                    break
        
        return word_probs, word_alphas

    def generate_square_subsequent_mask(self, sz):
        """Generate causal mask for Transformer decoder."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_hidden(self, features, feature_mask):
        # Retained for compatibility, but Transformer doesn't use hidden; return dummy
        average = (features * feature_mask).sum(-1).sum(-1) / feature_mask.sum(-1).sum(-1)
        return torch.tanh(average)  # Not used in forward; can remove if not needed elsewhere
