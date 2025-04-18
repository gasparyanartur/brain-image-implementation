from pydantic import BaseModel


class BaseConfig(BaseModel):
    config_tag: str

    @classmethod
    def from_hydra_config(cls, cfg):
        """Create an instance of the config class from a Hydra config."""
        return cls(**cfg)
