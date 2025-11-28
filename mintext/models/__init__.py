import mlxu
from mintext.models.llama_model import LLaMAModel, LLaMAConfigurator, LLaMAShardingConfig


class ModelConfigurator(object):
    """ Configurator to support multiple different model types. """

    @staticmethod
    def get_default_config(updates=None):
        config = mlxu.config_dict()
        config.model_type = 'llama'
        config.llama_model = LLaMAConfigurator.get_default_config()
        config.llama_sharding = LLaMAShardingConfig.get_default_config()

        return mlxu.update_config_dict(config, updates)

    @classmethod
    def make_model_and_sharding(cls, config, dtype, param_dtype):
        config = cls.get_default_config(config)

        if config.model_type == 'llama':
            model_config = LLaMAConfigurator.finalize_config(config.llama_model)
            sharding = LLaMAShardingConfig(config.llama_sharding)

            model = LLaMAModel(
                model_config,
                dtype=dtype,
                param_dtype=param_dtype,
            )
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

        return model_config, model, sharding
