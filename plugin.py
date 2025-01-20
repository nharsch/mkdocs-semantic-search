# mkdocs_semantic_search/plugin.py
import os
import json
from mkdocs.plugins import BasePlugin
from mkdocs.config import config_options
from mkdocs.structure.pages import Page
from mkdocs.structure.files import Files

class SemanticSearchPlugin(BasePlugin):
    config_scheme = (
        ('enabled', config_options.Type(bool, default=True)),
        ('embedding_file', config_options.Type(str, default='embeddings.json')),
    )

    def on_config(self, config):
        """Called when the config is loaded."""
        if not self.config['enabled']:
            return config
        
        # Add plugin static assets directory to the static_templates
        static_path = os.path.join(os.path.dirname(__file__), 'static')
        config['theme'].dirs.append(static_path)
        
        return config

    def on_files(self, files: Files, config):
        """Called after the files are loaded."""
        if not self.config['enabled']:
            return files
        
        # Add our JS files to the build
        src_path = os.path.join(os.path.dirname(__file__), 'static')
        
        files.append_files([
            files.File(
                path='js/semantic-search.js',
                src_dir=src_path,
                dest_dir=config['site_dir'],
                use_directory_urls=False
            )
        ])
        
        return files

    def on_page_context(self, context, page: Page, config, nav):
        """Add plugin config to the page context."""
        if not self.config['enabled']:
            return context
        
        context['semantic_search'] = {
            'enabled': True,
            'embedding_file': self.config['embedding_file']
        }
        
        return context

    def on_post_build(self, config):
        """Generate embeddings after the site is built."""
        if not self.config['enabled']:
            return
        
        # Create embeddings.js file in the built site
        output_dir = config['site_dir']
        embedding_path = os.path.join(output_dir, 'js', self.config['embedding_file'])
        
        # Ensure the js directory exists
        os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
        
        # Copy our node script to a temporary location and run it
        self._generate_embeddings(config['docs_dir'], embedding_path)

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

# mkdocs_semantic_search/static/js/semantic-search.js
document.addEventListener('DOMContentLoaded', async function() {
    if (!window.semanticSearch?.enabled) {
        return;
    }

    // Load the embeddings
    const response = await fetch(`/js/${window.semanticSearch.embedding_file}`);
    const embeddings = await response.json();
    
    // Import transformers.js and initialize the model
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.14.0/+esm');
    
    const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        quantized: false
    });

    // Add the search UI
    const searchContainer = document.createElement('div');
    searchContainer.innerHTML = `
        <div class="semantic-search">
            <input type="text" placeholder="Search..." id="semantic-search-input">
            <div id="semantic-search-results"></div>
        </div>
    `;
    
    document.querySelector('.md-header__inner').appendChild(searchContainer);
    
    // Add search functionality
    const searchInput = document.getElementById('semantic-search-input');
    const resultsContainer = document.getElementById('semantic-search-results');
    
    searchInput.addEventListener('input', async function(e) {
        const query = e.target.value;
        if (!query) {
            resultsContainer.innerHTML = '';
            return;
        }
        
        // Generate embedding for the query
        const queryEmbedding = await embedder(query, {
            pooling: 'mean',
            normalize: true
        });
        
        // Compute similarities and sort results
        const results = Object.entries(embeddings)
            .map(([path, embedding]) => ({
                path,
                similarity: cosineSimilarity(queryEmbedding.data, embedding)
            }))
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, 5);
            
        // Display results
        resultsContainer.innerHTML = results
            .map(result => `
                <a href="/${result.path}" class="search-result">
                    <div>${result.path}</div>
                    <div>Score: ${result.similarity.toFixed(2)}</div>
                </a>
            `)
            .join('');
    });
});

function cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}

# mkdocs_semantic_search/static/css/semantic-search.css
.semantic-search {
    position: relative;
    margin: 1rem;
}

.semantic-search input {
    width: 100%;
    padding: 0.5rem;
    border: 1px solid #ccc;
    border-radius: 4px;
}

#semantic-search-results {
    position: absolute;
    top: 100%;
    left: 0;
    right: 0;
    background: white;
    border: 1px solid #ccc;
    border-radius: 4px;
    margin-top: 0.5rem;
    max-height: 400px;
    overflow-y: auto;
    display: none;
}

#semantic-search-results:not(:empty) {
    display: block;
}

.search-result {
    padding: 0.5rem;
    border-bottom: 1px solid #eee;
    display: block;
    color: inherit;
    text-decoration: none;
}

.search-result:hover {
    background: #f5f5f5;
}