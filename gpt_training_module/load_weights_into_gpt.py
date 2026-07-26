import numpy as np
import torch
import torch.nn

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, "
                          "Right: {right.shape}"
        )

    # `torch.nn.Parameter` has two properties `weight` & `bias`.
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for trf_block in range(len(params["blocks"])):

        # Splitting one giant matrix into three smaller attention's weight matrices q_w, k_w, v_w.
        q_w, k_w, v_w = np.split((params["blocks"][trf_block]["attn"]["c_attn"])["w"], 3, axis=-1)

        # Splitting one giant matrix into three smaller attention's biases matrices q_b, k_b, v_b.
        q_b, k_b, v_b = np.split((params["blocks"][trf_block]["attn"]["c_attn"])["b"], 3, axis=-1)

        # Loading values of `W_query`, `W_key`, `W_value` in each transformer block's attention mechanism's `nn.Embedding` layer's weight parameter.
        gpt.trf_blocks[trf_block].att.W_query.weight = assign(gpt.trf_blocks[trf_block].att.W_query.weight, q_w.T)
        gpt.trf_blocks[trf_block].att.W_key.weight = assign(gpt.trf_blocks[trf_block].att.W_key.weight, k_w.T)
        gpt.trf_blocks[trf_block].att.W_value.weight = assign(gpt.trf_blocks[trf_block].att.W_value.weight, v_w.T)

        # Loading values of `W_bias`, `W_bias`, `W_bias` in each transformer block's attention mechanism's `nn.Embedding` layer's weight parameter.
        gpt.trf_blocks[trf_block].att.W_query.bias = assign(gpt.trf_blocks[trf_block].att.W_query.bias, q_b)
        gpt.trf_blocks[trf_block].att.W_key.bias = assign(gpt.trf_blocks[trf_block].att.W_key.bias, k_b)
        gpt.trf_blocks[trf_block].att.W_value.bias = assign(gpt.trf_blocks[trf_block].att.W_value.bias, v_b)

        # Loading values of `att.out_proj.weight` and `att.out_proj.bias` in each transformer block's attention mechanism.
        gpt.trf_blocks[trf_block].att.out_proj.weight = assign(gpt.trf_blocks[trf_block].att.out_proj.weight, params["blocks"][trf_block]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[trf_block].att.out_proj.bias = assign(gpt.trf_blocks[trf_block].att.out_proj.bias, params["blocks"][trf_block]["attn"]["c_proj"]["b"])

        # Loading values of weights & bias of first layer of feedforward module in each transformer block.
        gpt.trf_blocks[trf_block].ff.layers[0].weight = assign(gpt.trf_blocks[trf_block].ff.layers[0].weight, params["blocks"][trf_block]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[trf_block].ff.layers[0].bias = assign(gpt.trf_blocks[trf_block].ff.layers[0].bias, params["blocks"][trf_block]["mlp"]["c_fc"]["b"])

        # Loading values of weights & bias of second layer of feedforward module in each transformer block.
        gpt.trf_blocks[trf_block].ff.layers[2].weight = assign(gpt.trf_blocks[trf_block].ff.layers[2].weight, params["blocks"][trf_block]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[trf_block].ff.layers[2].bias = assign(gpt.trf_blocks[trf_block].ff.layers[2].bias, params["blocks"][trf_block]["mlp"]["c_proj"]["b"])

        # Loading values of scale & shift of first normalization layer in each transformer block.
        gpt.trf_blocks[trf_block].norm1.scale = assign(gpt.trf_blocks[trf_block].norm1.scale, params["blocks"][trf_block]["ln_1"]["g"])
        gpt.trf_blocks[trf_block].norm1.shift = assign(gpt.trf_blocks[trf_block].norm1.shift, params["blocks"][trf_block]["ln_1"]["b"])

        # Loading values of scale & shift of second normalization layer in each transformer block.
        gpt.trf_blocks[trf_block].norm2.scale = assign(gpt.trf_blocks[trf_block].norm2.scale, params["blocks"][trf_block]["ln_2"]["g"])
        gpt.trf_blocks[trf_block].norm2.shift = assign(gpt.trf_blocks[trf_block].norm2.shift, params["blocks"][trf_block]["ln_2"]["b"])

    # Loading values of scale & shift of normalization layer in GPT model after transformer blocks.
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])

    # Loading values of output layer (projecting embeddings to vocabulary dimensions) in GPT model.
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])