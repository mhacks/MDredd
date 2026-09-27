import os

# Settings are validated on import and require a token.
os.environ.setdefault("MDREDD_API_TOKEN", "test-token-" + "x" * 32)
