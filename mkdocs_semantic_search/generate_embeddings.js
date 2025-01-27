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

    extractSections(content) {
        const sections = [];
        let currentHeader = null;
        let currentContent = [];

        const tokens = marked.lexer(content);

        tokens.forEach(token => {
            if (token.type === 'heading') {
                // Save previous section if exists
                if (currentHeader && currentContent.length) {
                    sections.push({
                        header: currentHeader,
                        content: this.cleanText(currentContent.join('\n'))
                    });
                    currentContent = [];
                }

                // Start new section
                currentHeader = token.text;
            } else if (currentHeader) {
                // Add token to current section
                currentContent.push(marked.parser([token]));
            }
        });

        // Add last section
        if (currentHeader && currentContent.length) {
            sections.push({
                header: currentHeader,
                content: this.cleanText(currentContent.join('\n'))
            });
        }

        return sections;
    }

    async embeddContent(content) {
        return await this.embedder(content, {
            pooling: 'mean',
            normalize: true
        });
    }

    cleanText(text) {
        return text
            .replace(/<[^>]*>/g, '') // Remove HTML tags
            .replace(/\s+/g, ' ') // Normalize whitespace
            .trim();
    }

    async embeddSections(content) {
        // Remove frontmatter
        const { body } = frontMatter(content);

        // Extract sections
        const sections = this.extractSections(body);

        for (const section of sections) {
            const text = this.cleanText(section.content);
            // Skip empty sections
            if (!text || text.length < 5) {
                // console.log('skipping short section:', text);
                continue;
            }
            // cast Float32Array to Array to help with serialization
            const embedding = await this.embeddContent(text);
            section.embedding = Array.from(embedding.data);
        }

        // Remove HTML tags and normalize whitespace
        return sections;
    }

    async processDocument(filePath, docsDir) {
        const content = await fs.readFile(filePath, 'utf-8');
        const embeddedSections = await this.embeddSections(content);

        // Get relative path from docs directory
        const relativePath = path.relative(docsDir, filePath);
        
        return {
            path: relativePath,
            embeddings: embeddedSections
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
            // generate embedding for each section of the markdown file

            const { path, embeddings: sectionEmbeddings } = await this.processDocument(file, docsDir);
            embeddings[path] = sectionEmbeddings;
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
