"""Silence Hugging Face's progress bars. Import this BEFORE transformers or huggingface_hub.

Outside a TTY -- a notebook cell, a redirected log -- tqdm cannot rewrite its line, so one bar
becomes hundreds of "Loading weights: 57%|..." lines that bury everything else. The environment
variables have to be set before huggingface_hub reads them at import time, which is why this is
a module whose import does the work rather than a function someone might call too late.

Set HASTIKA_HF_VERBOSE=1 to get the bars back.
"""
import os

_QUIET = not os.environ.get("HASTIKA_HF_VERBOSE")
if _QUIET:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")     # the download bar
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows-only noise

from transformers.utils import logging as _hf_logging  # noqa: E402  (must follow the env setup)

if _QUIET:
    _hf_logging.disable_progress_bar()      # the "Loading weights" bar; verbosity is left alone
