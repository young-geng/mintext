import mlxu


class LLaMAConfigurator(object):
    """ Configurator for LLaMA 1,2,3 models. """

    @classmethod
    def get_default_config(cls, updates=None):
        config = mlxu.config_dict()
        config.base_model = 'llama3_8b'
        config.vocab_size = mlxu.config_placeholder(int)
        config.hidden_size = mlxu.config_placeholder(int)
        config.intermediate_size = mlxu.config_placeholder(int)
        config.num_hidden_layers = mlxu.config_placeholder(int)
        config.num_attention_heads = mlxu.config_placeholder(int)
        config.num_key_value_heads = mlxu.config_placeholder(int)
        config.initializer_scale = mlxu.config_placeholder(float)
        config.rms_norm_eps = mlxu.config_placeholder(float)
        config.max_position_embeddings = mlxu.config_placeholder(int)
        config.rope_theta = mlxu.config_placeholder(float)
        config.embedding_dropout = mlxu.config_placeholder(float)
        config.feedforward_dropout = mlxu.config_placeholder(float)
        config.attention_dropout = mlxu.config_placeholder(float)
        config.residue_dropout = mlxu.config_placeholder(float)
        config.remat = mlxu.config_placeholder(str)
        config.attention_chunk_size = mlxu.config_placeholder(int)
        return mlxu.update_config_dict(config, updates)

    @classmethod
    def finalize_config(cls, config):
        """ Apply updates on top of standard base model config. """
        standard_config = cls.get_standard_llama_config(config.base_model)
        for key, value in config.items():
            if key != 'base_model' and value is not None:
                standard_config[key] = value
        return standard_config

    @classmethod
    def get_standard_llama_config(cls, model_name):
        config = mlxu.config_dict()
        config.base_model = 'llama_7b'
        config.vocab_size = 32000
        config.hidden_size = 4096
        config.intermediate_size = 11008
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.num_key_value_heads = 32
        config.initializer_scale = 1.0
        config.rms_norm_eps = 1e-6
        config.max_position_embeddings = 2048
        config.rope_theta = 1e4
        config.embedding_dropout = 0.0
        config.feedforward_dropout = 0.0
        config.attention_dropout = 0.0
        config.residue_dropout = 0.0
        config.remat = 'block'
        config.attention_chunk_size = 1024

        updates = {
            'debug': dict(
                base_model='debug',
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                rms_norm_eps=1e-6,
            ),
            'llama_1b': dict(
                base_model='llama_1b',
                hidden_size=2048,
                intermediate_size=5504,
                num_hidden_layers=22,
                num_attention_heads=16,
                num_key_value_heads=16,
                rms_norm_eps=1e-6,
            ),
            'llama_3b': dict(
                base_model='llama_3b',
                hidden_size=3200,
                intermediate_size=8640,
                num_hidden_layers=26,
                num_attention_heads=32,
                num_key_value_heads=32,
                rms_norm_eps=1e-6,
            ),
            'llama_7b': dict(
                base_model='llama_7b',
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                rms_norm_eps=1e-6,
            ),
            'llama_13b': dict(
                base_model='llama_13b',
                hidden_size=5120,
                intermediate_size=13824,
                num_hidden_layers=40,
                num_attention_heads=40,
                num_key_value_heads=40,
                rms_norm_eps=1e-6,
            ),
            'llama_30b': dict(
                base_model='llama_30b',
                hidden_size=6656,
                intermediate_size=17920,
                num_hidden_layers=60,
                num_attention_heads=52,
                num_key_value_heads=52,
                rms_norm_eps=1e-6,
            ),
            'llama_65b': dict(
                base_model='llama_65b',
                hidden_size=8192,
                intermediate_size=22016,
                num_hidden_layers=80,
                num_attention_heads=64,
                num_key_value_heads=64,
                rms_norm_eps=1e-5,
            ),
            'llama2_7b': dict(
                base_model='llama2_7b',
                hidden_size=4096,
                intermediate_size=11008,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=32,
                max_position_embeddings=4096,
                rms_norm_eps=1e-5,
            ),
            'llama2_13b': dict(
                base_model='llama2_13b',
                hidden_size=5120,
                intermediate_size=13824,
                num_hidden_layers=40,
                num_attention_heads=40,
                num_key_value_heads=40,
                max_position_embeddings=4096,
                rms_norm_eps=1e-5,
            ),
            'llama2_70b': dict(
                base_model='llama_65b',
                hidden_size=8192,
                intermediate_size=28672,
                num_hidden_layers=80,
                num_attention_heads=64,
                num_key_value_heads=8,
                max_position_embeddings=4096,
                rms_norm_eps=1e-5,
            ),
            'llama3_8b': dict(
                base_model='llama3_8b',
                vocab_size=128256,
                hidden_size=4096,
                intermediate_size=14336,
                num_hidden_layers=32,
                num_attention_heads=32,
                num_key_value_heads=8,
                max_position_embeddings=8192,
                rms_norm_eps=1e-5,
                rope_theta=5e5,
            ),
            'llama3_70b': dict(
                base_model='llama3_8b',
                vocab_size=128256,
                hidden_size=8192,
                intermediate_size=28672,
                num_hidden_layers=80,
                num_attention_heads=64,
                num_key_value_heads=8,
                max_position_embeddings=8192,
                rms_norm_eps=1e-5,
                rope_theta=5e5,
            ),
        }[model_name]
        return mlxu.update_config_dict(config, updates)
