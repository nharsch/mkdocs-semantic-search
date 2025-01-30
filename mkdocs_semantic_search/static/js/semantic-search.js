document.addEventListener('DOMContentLoaded', async function() {
  // Load the embeddings
  // TODO: load or template in embeddings
  const response = await fetch(`/semantic-search/embeddings.json`);
  const embeddings = await response.json();
  console.log("embeddings: ", embeddings)

  // Import transformers.js and initialize the model
  // TODO: can / should we install this and load it in
  const { pipeline } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.3.2');
  // console.log("pipeline: ", pipeline)

  const embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    quantized: false
  });

  console.log("embedder: ", embedder)


  // Add the toggle switch next to the search box
  const navBar = document.querySelector('.md-header__inner');
  const searchContainer = document.querySelector('.md-search');
  const toggleSwitch = document.createElement('div');
  toggleSwitch.innerHTML = `
        <label class="md-switch">
            <span class="md-switch__thumb">
                <span class="md-switch__track">AI Powered Search</span>
            </span>
            <input type="checkbox" id="semantic-search-toggle">
        </label>
    `;
  navBar.insertBefore(toggleSwitch, searchContainer);

  // Intercept the search input
  const searchInput = document.querySelector('.md-search__input');
  let isSemanticSearch = false;

  // Add the event listener for the toggle switch
  document.getElementById('semantic-search-toggle').addEventListener('change', (e) => {
    isSemanticSearch = e.target.checked;
    console.log("toggle switch changed, isSemanticSearch: ", isSemanticSearch);
  });

  // TODO: implement debounce on search results
  // const debounce = (func, wait) => {
  //   let timeout;
  //   return function executedFunction(...args) {
  //     const later = () => {
  //       clearTimeout(timeout);
  //       func(...args);
  //     };
  //     clearTimeout(timeout);
  //     timeout = setTimeout(later, wait);
  //   };
  // };


  // hijack the search input event
  searchInput.addEventListener('input', async function(e) {
    const searchContainer = document.querySelector('.md-search-result');
    const searchMeta = searchContainer.querySelector('.md-search-result__meta');
    const searchList = searchContainer.querySelector('.md-search-result__list');

    if (!isSemanticSearch) {
      // Let the default search handle it
      console.log("setting default search display back to block");
      if (searchMeta.style.display === 'none') {
        searchMeta.style.display = 'block';
      }
      if (searchList.style.display === 'none') {
        searchList.style.display = 'block';
      }
      return;
    }

    console.log("semantic search input event");

    // clear the meta and list elements
    console.log("hiding default results dom elements");
    searchMeta.style.display = 'none';
    searchList.style.display = 'none';

    // build the semantic results container
    resultsContainer = searchContainer.querySelector('.semantic-search-results');
    if (!resultsContainer) {
        const resultsContainer = document.createElement('ol');
        resultsContainer.classList.add('md-search-result__list');
        resultsContainer.classList.add('semantic-search-results');
        searchContainer.appendChild(resultsContainer);
    }

    const query = e.target.value;

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

    console.log("found results: ", results);

    // Display results
    resultsContainer.innerHTML = results
      .map(result => {
        // TODO: remove this hacky solution
        const htmlPath = result.path.replace(".md", ".html");
        const header = slugify(result.header)
        const fullPath = `${htmlPath}#${header}`;
        return `<li class="md-search-result__item">
                    <a href="/${fullPath}" class="md-search-result__link">
                        <article class="md-search-result__article md-typeset"/>
                            <div class="md-search-result__icon md-icon"></div>
                            <h1>${result.path}</h1>
                        </article>
                    </a>
                    <a href="/${fullPath}" class="md-search-result__link">
                        <article class="md-search-result__article md-typeset"/>
                            <h2>${result.header}</h2>
                            <p>${result.content}</p>
                            <p>Score: ${result.similarity.toFixed(2)}</p>
                        </article>
                    </a>
                </li>`
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
