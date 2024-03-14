"""T5-based model congfiguration"""

from configuration_t5 import T5Config


class T5ConfigAdapter(T5Config):
    def __init__(self, train_adapters=False, **kwargs):
        super().__init__(**kwargs)
        self.train_adapters = train_adapters