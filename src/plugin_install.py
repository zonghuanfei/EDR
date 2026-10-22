"""Attach EDR to the MA-RAG runner without editing the upstream repository."""

from edr import Plugin


def install_edr(harness_module, label_token_ids, first_round_sources=None):
    plugin = Plugin(harness_module, label_token_ids, first_round_sources or {})
    plugin.install()
    return plugin
