# Debate RL

This repository contains a minimal multi-turn debate rollout compatible with the OpenPipe ART
interface. LLM1 is assumed to be trainable through an ART `art.Model` instance, while LLM2 and the
judge are configured with external chat-completions clients (OpenAI or Hugging Face compatible).
All messaging structures mirror the ART notebook expectations so this package can slot into the
broader training flow once real backends are supplied.

## Quick start

```bash
pip install -e .
python -m examples.run_smoke_test
pytest -q
```

The bundled smoke test and pytest suite rely on lightweight stub clients that satisfy the same
interfaces as real chat-completion APIs. Swap those stubs with production clients when integrating
with live models.
