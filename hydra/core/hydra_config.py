class HydraConfig:
    _cfg = None

    def __call__(self):
        return self

    @classmethod
    def set_config(cls, cfg):
        cls._cfg = cfg

    @classmethod
    def get(cls):
        if cls._cfg is None:
            raise ValueError("HydraConfig is not set")
        return cls._cfg
