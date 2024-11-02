from functools import partial
from typing import Callable

import numpy as np
import mlxu
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as PS
from jax.experimental.shard_map import shard_map
import flax
import flax.linen as nn
import einops
from ringattention import ringattention

from scalax.sharding import (
    MeshShardingHelper, TreePathShardingRule, with_sharding_annotation
)


class LLaMAShardingConfig(object):
    """Sharding config for llama model."""

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.mesh_dim = '1,-1,1,1'
        config.shard_model_along_sequence = False
        return mlxu.update_config_dict(config, updates)

    def __init__(self, config):
        self.config = self.get_default_config(config)
        self._ring_attention_function = None

    def get_mesh(self):
        axis_dims = self.config.mesh_dim
        if axis_dims.startswith('!'):
            # Allow splitting a physical mesh axis if needed
            mesh_axis_splitting = True
            axis_dims = axis_dims[1:]
        else:
            mesh_axis_splitting = False

        names = ('replica', 'fsdp', 'sequence', 'tensor')
        dims = [int(x) for x in axis_dims.split(',')]
        assert len(dims) == len(names)
        return MeshShardingHelper(dims, names, mesh_axis_splitting)

    def get_model_sharding_rule(self):
        """ Get the tree path based partition rule for LLaMA model. """
        if self.config.shard_model_along_sequence:
            model_all_gather_axis = ('fsdp', 'sequence')
        else:
            model_all_gather_axis = 'fsdp'
        return TreePathShardingRule(
            # embeddings
            ('transformer/wte/embedding', PS('tensor', model_all_gather_axis)),
            # atention
            ('self_attention/(k_proj|q_proj|v_proj)/kernel', PS(model_all_gather_axis, 'tensor')),
            ('self_attention/o_proj/kernel', PS('tensor', model_all_gather_axis)),
            # mlp
            ('feedforward/up_proj/kernel', PS(model_all_gather_axis, 'tensor')),
            ('feedforward/down_proj/kernel', PS('tensor', model_all_gather_axis)),
            ('feedforward/gate_proj/kernel', PS(model_all_gather_axis, 'tensor')),
            # layer norms
            ('input_layernorm/scale', PS(None)),
            ('post_attention_layernorm/scale', PS(None)),
            # output head
            ('lm_head_norm/scale', PS(None)),
            ('lm_head/kernel', PS(model_all_gather_axis, 'tensor')),
            ('.*', PS(None)),
        )

    def get_intermediate_sharding_rules(self):
        return {
            'data': PS(('replica', 'fsdp'), 'sequence'),
            'ffw_intermediate': PS(('replica', 'fsdp'), 'sequence', 'tensor'),
            'attention_kqv': PS(('replica', 'fsdp'), 'sequence', 'tensor'),
            'mask': PS(('replica', 'fsdp'), 'sequence'),
        }

    def get_batch_sharding(self):
        return PS(('replica', 'fsdp'), 'sequence')


def get_ring_attention_function(
    chunk_size,
    deterministic=True,
    attention_dropout=0.0,
    dropout_rng=None,
):
    return shard_map(
        partial(
            ringattention,
            axis_name='sequence',
            float32_logits=True,
            cache_idx=None,
            blockwise_kwargs=dict(
                causal_block_size=1,
                deterministic=deterministic,
                dropout_rng=dropout_rng,
                attn_pdrop=attention_dropout,
                query_chunk_size=chunk_size,
                key_chunk_size=chunk_size,
                policy=jax.checkpoint_policies.nothing_saveable,
                dtype=jnp.float32,
                precision=None,
                prevent_cse=True,
            )
        ),
        mesh=MeshShardingHelper.get_global_mesh(),
        in_specs=(
            PS(('replica', 'fsdp'), 'sequence', 'tensor', None),
            PS(('replica', 'fsdp'), 'sequence', 'tensor', None),
            PS(('replica', 'fsdp'), 'sequence', 'tensor', None),
            PS(('replica', 'fsdp'), None, None, None),
            PS(('replica', 'fsdp'), None),
        ),
        out_specs=PS(('replica', 'fsdp'), 'sequence', 'tensor', None),
        check_rep=False
    )


def apply_rotary_emb(xq, xk, position_ids, max_pos, theta=10000.0):
    input_dtype = xq.dtype
    with jax.ensure_compile_time_eval():
        dim = xq.shape[-1]
        freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
        t = jnp.arange(max_pos)
        freqs = jnp.outer(t, freqs).astype(jnp.float32)
        sin, cos = jnp.sin(freqs), jnp.cos(freqs)
        freqs_cis = jnp.complex64(cos + 1j * sin)
    freqs_cis = jnp.take(freqs_cis, position_ids, axis=0)
    reshape_xq = xq.astype(jnp.float32).reshape(*xq.shape[:-1], -1, 2)
    reshape_xk = xk.astype(jnp.float32).reshape(*xk.shape[:-1], -1, 2)

    xq_ = jax.lax.complex(reshape_xq[..., 0], reshape_xq[..., 1])
    xk_ = jax.lax.complex(reshape_xk[..., 0], reshape_xk[..., 1])
    # add head dim
    freqs_cis = jnp.reshape(freqs_cis, (*freqs_cis.shape[:2], 1, *freqs_cis.shape[2:]))
    xq_out = xq_ * freqs_cis
    xq_out = jnp.stack((jnp.real(xq_out), jnp.imag(xq_out)), axis=-1).reshape(*xq_out.shape[:-1], -1)
    xk_out = xk_ * freqs_cis
    xk_out = jnp.stack((jnp.real(xk_out), jnp.imag(xk_out)), axis=-1).reshape(*xk_out.shape[:-1], -1)
    return xq_out.astype(input_dtype), xk_out.astype(input_dtype)


class FeedForward(nn.Module):
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, deterministic=True):
        w1 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(self.config.hidden_size),
                dtype=self.param_dtype
            ),
            name='gate_proj',
        )
        w2 = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(self.config.intermediate_size),
                dtype=self.param_dtype
            ),
            name='down_proj',
        )
        w3 = nn.Dense(
            self.config.intermediate_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(self.config.hidden_size),
                dtype=self.param_dtype
            ),
            name='up_proj',
        )
        x = w2(
            nn.silu(with_sharding_annotation(w1(x), 'ffw_intermediate'))
            * with_sharding_annotation(w3(x), 'ffw_intermediate')
        )
        x = with_sharding_annotation(x, 'ffw_output')
        return nn.Dropout(rate=self.config.feedforward_dropout)(x, deterministic=deterministic)


class Attention(nn.Module):
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, attention_mask, position_ids, segment_ids, deterministic=True):
        assert self.config.hidden_size % self.config.num_key_value_heads == 0
        assert self.config.hidden_size % self.config.num_attention_heads == 0

        sequence_length = hidden_states.shape[1]
        init_scale = self.config.initializer_range / np.sqrt(self.config.hidden_size)
        num_query_groups = self.config.num_attention_heads // self.config.num_key_value_heads

        xq = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(init_scale),
            name='q_proj',
        )(hidden_states)
        xk = nn.Dense(
            self.config.hidden_size // num_query_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(init_scale),
            name='k_proj',
        )(hidden_states)
        xv = nn.Dense(
            self.config.hidden_size // num_query_groups,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(init_scale),
            name='v_proj',
        )(hidden_states)

        xq = einops.rearrange(
            xq, 'b s (h d) -> b s h d',
            h=self.config.num_attention_heads,
        )
        xk = einops.repeat(
            xk, 'b s (h d) -> b s (h g) d',
            h=self.config.num_key_value_heads,
            g=num_query_groups,
        )
        xv = einops.repeat(
            xv, 'b s (h d) -> b s (h g) d',
            h=self.config.num_key_value_heads,
            g=num_query_groups,
        )

        xq = with_sharding_annotation(xq, 'attention_kqv')
        xk = with_sharding_annotation(xk, 'attention_kqv')
        xv = with_sharding_annotation(xv, 'attention_kqv')

        xq, xk = apply_rotary_emb(
            xq, xk, position_ids,
            max_pos=self.config.max_position_embeddings,
            theta=self.config.rope_theta,
        )

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng('dropout')

        attention_bias = jax.lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, jnp.finfo(self.dtype).min).astype(self.dtype),
        )
        attention_bias = einops.rearrange(attention_bias, 'b s -> b 1 1 s')
        attention_output = get_ring_attention_function(
            chunk_size=self.config.attention_chunk_size,
            deterministic=deterministic,
            attention_dropout=self.config.attention_dropout,
            dropout_rng=dropout_rng,
        )(xq, xk, xv, attention_bias, segment_ids).astype(self.dtype)

        attention_output = einops.rearrange(attention_output, 'b s h d -> b s (h d)')

        x_out = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(init_scale),
            name='o_proj',
        )(attention_output)

        x_out = nn.Dropout(rate=self.config.residue_dropout)(
            x_out, deterministic=deterministic
        )
        return x_out


class TransformerBlock(nn.Module):
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, hidden_states, attention_mask, position_ids,
                 segment_ids, deterministic=True):
        x_out = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            param_dtype=self.param_dtype,
            name='input_layernorm',
        )(hidden_states)
        x_out = Attention(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='self_attention',
        )(x_out, attention_mask, position_ids, segment_ids, deterministic=deterministic)
        mlp_inputs = x_out + hidden_states
        x_out = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=jnp.promote_types(self.dtype, jnp.float32),
            param_dtype=self.param_dtype,
            name='post_attention_layernorm',
        )(mlp_inputs)
        x_out = FeedForward(
            config=self.config,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name='feedforward',
        )(x_out, deterministic=deterministic)
        return x_out + mlp_inputs


class LLaMAModel(nn.Module):
    config: mlxu.ConfigDict
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @classmethod
    @nn.nowrap
    def rng_keys(cls):
        return ('params', 'dropout')

    @nn.compact
    def __call__(self, input_ids, attention_mask, position_ids, segment_ids, deterministic=True):
        hidden_states = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                self.config.initializer_range,
                dtype=self.param_dtype
            ),
            dtype=self.param_dtype,
            name='embeddings',
        )(input_ids)
        hidden_states = nn.Dropout(
            rate=self.config.embedding_dropout, name='emb_drop'
        )(hidden_states, deterministic=deterministic)

        remat_policy = {
            'block': jax.checkpoint_policies.nothing_saveable,
            'dots': jax.checkpoint_policies.checkpoint_dots,
            'dots_with_no_batch_dims': jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
            'none': jax.checkpoint_policies.everything_saveable,
        }
        block_module = nn.remat(
            TransformerBlock,
            policy=remat_policy[self.config.remat],
            static_argnums=(5,),
        )

        for i in range(self.config.num_hidden_layers):
            hidden_states = block_module(
                self.config,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f'transformer_block_{i}',
            )(hidden_states, attention_mask, position_ids, segment_ids, deterministic)

        hidden_states = nn.RMSNorm(
            epsilon=self.config.rms_norm_eps,
            dtype=jnp.float32,
            param_dtype=self.param_dtype,
            name='lm_head_norm',
        )(hidden_states)
        logits = nn.remat(nn.Dense)(
            self.config.vocab_size,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range / np.sqrt(self.config.hidden_size),
                dtype=self.param_dtype
            ),
            name='lm_head',
        )(hidden_states)
        return logits
