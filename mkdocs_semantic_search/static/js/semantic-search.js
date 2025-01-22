document.addEventListener('DOMContentLoaded', async function() {
    // TODO: where do I set this?
    // if (!window.semanticSearch?.enabled) {
    //     return;
    // }

    // Load the embeddings
    // TODO: load or template in embeddings
    const response = await fetch(`/semantic-search/embeddings.json`);
    const embeddings = await response.json();
    console.log("embeddings: ", embeddings)

    // Import transformers.js and initialize the model
    // TODO: can / should we install this and load it in
    const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.2');
    console.log("pipeline: ", pipeline)

    const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
        quantized: false
    });

    console.log("embedder: ", embedder)

    // Add the search UI
    console.log("mounting search field")
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
            .map(result => {
               // TODO: remove this hacky solution
               const htmlPath = result.path.replace(".md", ".html");
               return `<a href="/${htmlPath}" class="search-result">
                    <div>${result.path}</div>
                    <div>Score: ${result.similarity.toFixed(2)}</div>
                </a>`
            })
            .join('');
    });
});

function cosineSimilarity(a, b) {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magnitudeA * magnitudeB);
}
