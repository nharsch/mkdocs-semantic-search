from mkdocs.config.config_options import (
    Choice,
    Deprecated,
    Optional,
    ListOfItems,
    Type
)
from mkdocs.config.base import Config
from mkdocs.contrib.search import LangOption


class SemanticSearchConfig(Config):
    enabled = Type(bool, default = True)
    embedding_file = Type(str, default = 'embeddings.json')
    # TODO: maybe set up a model arg?
    # TODO: what other options do we need?
