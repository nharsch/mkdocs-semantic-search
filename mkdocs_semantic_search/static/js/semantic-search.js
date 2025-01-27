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

    // TODO: add to regular search bar
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

    searchInput.addEventListener('keydown', async function(e) {
        if (e.key !== 'Enter') {
            return;
        }

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
        // embeddings is a list of { header, content, embedding }
              .flatMap(([path, sections]) => {
                  return sections.filter(section => !!section.embedding)
                                 .map((section) => {
                                     console.log("section: ", path, section.header);
                                     console.log("queryEmbedding: ", queryEmbedding.data)
                                     console.log("section.embedding: ", section.embedding)
                                     return {
                                         path: path,
                                         header: section.header,
                                         content: section.content,
                                         similarity: cosineSimilarity(queryEmbedding.data, section.embedding)
                                     }
                                 })
              })
              .sort((a, b) => b.similarity - a.similarity)
              .slice(0, 5);


        // Display results
        resultsContainer.innerHTML = results
            .map(result => {
               // TODO: remove this hacky solution
               const htmlPath = result.path.replace(".md", ".html");
               const header = slugify(result.header)
               const fullPath = `${htmlPath}#${header}`;
                return `<a href="/${fullPath}" class="search-result">
                    <div>${result.path}</div>
                    <div>${result.header}</div>
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

function slugify(str) {
  str = str.replace(/^\s+|\s+$/g, ''); // trim leading/trailing white space
  str = str.toLowerCase(); // convert string to lowercase
  str = str.replace(/[^a-z0-9 -]/g, '') // remove any non-alphanumeric characters
           .replace(/\s+/g, '-') // replace spaces with hyphens
           .replace(/-+/g, '-'); // remove consecutive hyphens
  return str;
}
