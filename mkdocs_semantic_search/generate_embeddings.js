import { pipeline } from '@xenova/transformers';
import { marked } from 'marked';
import frontMatter from 'front-matter';
import { glob } from 'glob';
import fs from 'fs/promises';
import path from 'path';

class DocEmbeddingsGenerator {
    constructor() {
        this.modelName = 'Xenova/all-MiniLM-L6-v2';
    }

    async initialize() {
        console.log('Loading embedding model...');
        this.embedder = await pipeline('feature-extraction', this.modelName, {
            quantized: false
        });
    }

    extractTextFromMarkdown(content) {
        // Remove frontmatter
        const { body } = frontMatter(content);
        
        // Convert markdown to plain text
        const text = marked.parse(body, { mangle: false, headerIds: false });
        
        // Remove HTML tags and normalize whitespace
        return text
            .replace(/<[^>]*>/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }

    async processDocument(filePath, docsDir) {
        const content = await fs.readFile(filePath, 'utf-8');
        const text = this.extractTextFromMarkdown(content);
        
        // Generate embedding
        const output = await this.embedder(text, {
            pooling: 'mean',
            normalize: true
        });
        
        // Get relative path from docs directory
        const relativePath = path.relative(docsDir, filePath);
        
        return {
            path: relativePath,
            embedding: Array.from(output.data)
        };
    }

    async generateEmbeddings(docsDir) {
        // Find all markdown files
        const files = await glob('**/*.md', {
            cwd: docsDir,
            absolute: true
        });

        console.log(`Found ${files.length} markdown files`);

        const embeddings = {};
        
        // Process each file
        for (const file of files) {
            console.log(`Processing ${file}`);
            const { path, embedding } = await this.processDocument(file, docsDir);
            embeddings[path] = embedding;
        }

        return embeddings;
    }

    async saveEmbeddings(embeddings, outputFile) {
        await fs.writeFile(
            outputFile,
            JSON.stringify(embeddings, null, 2)
        );
    }
}

// CLI script
async function main() {
    const docsDir = process.argv[2];
    const outputFile = process.argv[3] || 'embeddings.json';

    if (!docsDir) {
        console.error('Please provide the docs directory path');
        process.exit(1);
    }

    const generator = new DocEmbeddingsGenerator();
    await generator.initialize();

    console.log('Generating embeddings...');
    const embeddings = await generator.generateEmbeddings(docsDir);

    console.log(`Saving embeddings to ${outputFile}`);
    await generator.saveEmbeddings(embeddings, outputFile);
    
    console.log('Done!');
}

main().catch(console.error);
