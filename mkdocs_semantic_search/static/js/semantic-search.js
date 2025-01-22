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
