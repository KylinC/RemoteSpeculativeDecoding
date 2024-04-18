import os

def _env(name, value):
    if name in os.environ:
        return os.environ.get(name)
    else:
        return value

WS_SERVER_ADDR = _env("WS_SERVER_ADDR", "127.0.0.1")
WS_SERVER_PORT = _env("WS_SERVER_PORT", "2333")

MODEL_ZOO_DIR = _env("MODEL_ZOO_DIR", "/home/share")