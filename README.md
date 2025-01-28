# mkdocs Semantic Search Plugin

**Experimental** - this is a proff of concept inspired by 

An attempt to provide semantic search results using [transformers.js](https://github.com/huggingface/transformers.js-examples), inspired by [Semantic Finder](https://geo.rocks/semanticfinder/).

Requires [material for mkdocs](mkdocs-material) with default [search plugin](https://squidfunk.github.io/mkdocs-material/plugins/search/) enabled.

Documents embeddings are built at site build time. Query embeddings are built client side at query time.

Document embeddings are set at the section level, as I found more granular sentence / paragraph embeddings didn't lead to noticeable improvements.
