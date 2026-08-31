from utils.utils import Config

def model_config():
    config = Config({
        "slice_num": 5,
        "context_window": 5,
        "slice_ch": [16, 16, 32, 64, 192],
        "quant": "ste",
        "in_channels": 7,
        "out_channels": 7,
    })

    return config
