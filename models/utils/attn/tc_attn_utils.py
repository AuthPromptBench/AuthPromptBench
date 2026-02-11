'''
class AttentionControl and class AttentionStore are modified from
https://github.com/mlpc-ucsd/TokenCompose/blob/main/train/src/attn_utils.py
https://github.com/google/prompt-to-prompt/blob/main/prompt-to-prompt_stable.ipynb
https://github.com/google/prompt-to-prompt/blob/main/ptp_utils.py
https://github.com/huggingface/diffusers/blob/v0.34.0/src/diffusers/models/attention_processor.py#L3246
'''


import abc
import torch
import torch.nn.functional as F
import pdb


LOW_RESOURCE = False 
SD14_TO_SD21_RATIO = 1.5

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        attn = self.forward(attn, is_cross, place_in_unet)
        # attn: 8 * res * res

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def add_step(self):
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
 
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        # if is_cross:

        # only store cross attn 
        if is_cross and place_in_unet in self.train_layer_place:
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn.clone())
        
        return attn

    def between_steps(self):

        self.cur_step = 0
        self.attention_store = self.step_store
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        # debug: make sure the attn map will not be changed without clone
        average_attention = {key: [item for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, train_layer_ls):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        # only store training layer
        train_layer_place = []
        for inst in train_layer_ls:
            train_layer_place.append(inst.split('_')[0])
        self.train_layer_place = list(set(train_layer_place))

def register_attention_control(unet_model, controller):
    def ca_forward(self, place_in_unet, name):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out
        
        def forward(hidden_states, encoder_hidden_states=None, attention_mask=None,temb=None,):
            is_cross = encoder_hidden_states is not None
            
            residual = hidden_states

            # Spatial norm
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            # 2D -> sequence
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, self.heads, -1, attention_mask.shape[-1])
            # attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

            # Group norm
            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads

            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) 
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2) 

            # 🔥 核心：插入控制器
            if query.requires_grad or key.requires_grad or value.requires_grad or hidden_states.requires_grad:
                # if is_cross:
                attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
                if attention_mask is not None:
                    attn_scores = attn_scores + attention_mask
                attention_probs = F.softmax(attn_scores, dim=-1)
                attention_probs = attention_probs.reshape(-1, attention_probs.shape[2], attention_probs.shape[3]) # (batch_size * self.heads, seq_len, seq_len)
                controller(attention_probs, is_cross, place_in_unet)
            else:
                controller.add_step() 

            if self.norm_q is not None:
                query = self.norm_q(query)
            if self.norm_k is not None:
                key = self.norm_k(key)

            # the output of sdp = (batch, num_heads, seq_len, head_dim)
            # TODO: add support for attn.scale when we move to Torch 2.1
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
            hidden_states = hidden_states.to(query.dtype)

            # linear proj
            hidden_states = to_out(hidden_states)
            # all drop out in diffusers are 0.0
            # so we here ignore dropout

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states
            # print(query.shape, key.shape, value.shape)
            # query = self.head_to_batch_dim(query)
            # key = self.head_to_batch_dim(key)
            # value = self.head_to_batch_dim(value)
            # print(query.shape, key.shape, value.shape)
            # if attention_mask is not None:
            #     print(attention_mask.shape)
            # print('-----------------------------')

            # attention_probs = self.get_attention_scores(query, key, attention_mask)

            # # 🔥 核心：插入控制器
            # if attention_probs.requires_grad:
            #     attention_probs = controller(attention_probs, is_cross, place_in_unet)

            # hidden_states = torch.bmm(attention_probs, value)
            # hidden_states = self.batch_to_head_dim(hidden_states)

            # # linear proj
            # hidden_states = to_out(hidden_states)
            # # all drop out in diffusers are 0.0
            # # so we here ignore dropout

            # if input_ndim == 4:
            #     hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

            # if self.residual_connection:
            #     hidden_states = hidden_states + residual

            # hidden_states = hidden_states / self.rescale_output_factor

            # return hidden_states
        return forward


    assert controller is not None, "controller must be specified"
    num_att_layers = 0
    for name, module in unet_model.named_modules():
        if module.__class__.__name__ == 'Attention':
            module.__original_forward__ = module.forward
            if "down" in name:
                place_in_unet = "down"
            elif "mid" in name:
                place_in_unet = "mid"
            elif "up" in name:
                place_in_unet = "up"
            module.forward = ca_forward(module, place_in_unet, name)
            num_att_layers += 1
    controller.num_att_layers = num_att_layers

def get_cross_attn_map_from_unet(attention_store: AttentionStore, is_training_sd21, 
                           reses=[64, 32, 16, 8], poses=["down", "mid", "up"]):
    attention_maps = attention_store.get_average_attention()

    attn_dict = {}

    if is_training_sd21:
        reses = [int(SD14_TO_SD21_RATIO * item) for item in reses]
    # reses = [int(0.4375 * item) for item in reses]
    for pos in poses:
        for res in reses:
            temp_list = []
            for item in attention_maps[f"{pos}_cross"]:
                if item.shape[1] == res ** 2:
                    cross_maps = item.reshape(-1, res, res, item.shape[-1])
                    temp_list.append(cross_maps)
            # if such resolution exists
            if len(temp_list) > 0:
                attn_dict[f"{pos}_{res}"] = temp_list
    return attn_dict