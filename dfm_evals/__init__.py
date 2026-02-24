"""CLI and task extensions for inspect_ai."""

from .scorers import gleu
from .tasks import multi_wiki_qa

__all__ = ["multi_wiki_qa", "gleu"]
