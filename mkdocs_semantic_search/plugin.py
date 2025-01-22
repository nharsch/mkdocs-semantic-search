import os
import json
from mkdocs.plugins import BasePlugin
from mkdocs.config import config_options
from mkdocs.structure.pages import Page
from mkdocs.structure.files import Files, File

from .config import SemanticSearchConfig


class SemanticSearchPlugin(BasePlugin[SemanticSearchConfig]):
    config_scheme = (
        ('enabled', config_options.Type(bool, default=True)),
        ('embedding_file', config_options.Type(str, default='embeddings.json')),
    )

    def on_config(self, config):
        """Called when the config is loaded."""
        if not self.config.enabled:
            return config
        
        # Add plugin static assets directory to the static_templates
        # static_path = os.path.join(os.path.dirname(__file__), 'static')
        # config['theme'].dirs.append(static_path)
        
        return config

    def on_files(self, files: Files, config):
        """Called after the files are loaded."""
        if not self.config.enabled:
            return files
        
        # Add our JS files to the build
        src_path = os.path.join(os.path.dirname(__file__), 'static')

        print(f"writing files to {config.site_dir}")
        
        files.append(
            File(
                path='js/semantic-search.js',
                src_dir=src_path,
                dest_dir=config.site_dir,
                use_directory_urls=False
            )
        )
        
        return files

    def on_page_context(self, context, page: Page, config, nav):
        """Add plugin config to the page context."""
        if not self.config.enabled:
            return context
        
        context['semantic_search'] = {
            'enabled': True,
            'embedding_file': self.config.embedding_file
        }
        
        return context

    def on_post_build(self, config):
        """Generate embeddings after the site is built."""
        if not self.config.enabled:
            return
        
        # Create embeddings.js file in the built site
        output_dir = config.site_dir
        print(f"Output dir in embeddings config {output_dir}")
        embedding_path = os.path.join(output_dir, 'semantic-search', self.config.embedding_file)
        print(f"Generating embeddings to {embedding_path}")
        
        # Ensure the js directory exists
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        
        # Copy our node script to a temporary location and run it
        self._generate_embeddings(config.docs_dir, embedding_path)

    def _generate_embeddings(self, docs_dir: str, output_path: str):
        """Run the Node.js script to generate embeddings."""
        import subprocess
        import tempfile
        import shutil
        
        # Copy the Node.js script to a temporary location
        script_path = os.path.join(os.path.dirname(__file__), 'generate_embeddings.js')
        
        try:
            # Run the Node.js script
            subprocess.run([
                'node',
                script_path,
                docs_dir,
                output_path
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error generating embeddings: {e}")
            raise
