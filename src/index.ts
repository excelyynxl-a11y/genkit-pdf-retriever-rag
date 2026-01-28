import 'dotenv/config';
import { readFile } from 'fs/promises';
import path from 'path';

// import pdf from 'pdf-parse';
const { PDFParse } = require('pdf-parse');

import { ai, menuPdfIndexer } from './genkit';
import { z } from 'genkit';
import { chunk } from 'llm-chunk';
import { Document } from 'genkit/retriever';
import { devLocalIndexerRef } from '@genkit-ai/dev-local-vectorstore';
import googleAI from '@genkit-ai/googleai';

// plain async function to resolve pdf path and extract text from pdf
async function extractTextFromPdf(filePath: string) {
    const pdfFile = path.resolve(filePath);
    const dataBuffer = await readFile(pdfFile);
    // const data = await pdf(dataBuffer);

    const uint8Array = new Uint8Array(dataBuffer);
    const parser = new PDFParse(uint8Array);
	const data = await parser.getText();
	console.log(data.text);

    return data.text;
}

// chunking config function:
// guarantee a document segment of between 1000 and 2000 characters, 
// broken at the end of a sentence, 
// with an overlap between chunks of 100 characters
const chunkingConfig = {
    minLength: 1000,
    maxLength: 2000,
    splitter: 'sentence',
    overlap: 100,
    delimiters: '',
} as any;

// define the flow for indexing items in menu (?)
export const indexMenu = ai.defineFlow(
    {
        name: 'indexMenu', // name of flow
        inputSchema: z.object({ // input is a file path to a pdf
            filePath: z.string().describe('PDF file path'),
        }),
        outputSchema: z.object({ // output has succesfulness bool, item index, optionally error
            success: z.boolean(),
            documentsIndexed: z.number(),
            error: z.string().optional(),
        }),
    },
    async ({ filePath }) => {
        try {

            // resolve the file path
            filePath = path.resolve(filePath);

            // read the text in pdf via the extractTextFromPdf() method
            const pdfTxt = await ai.run('extract-text', () => extractTextFromPdf(filePath));

            // divide pdf text into segments
            const chunks = await ai.run('chunk-it', async () => chunk(pdfTxt, chunkingConfig));

            // convert chunks of text into documents to store in index
            const documents = chunks.map((text) => {
                return Document.fromText(text, { filePath })
            });

            // add document to index (refer to how we defined devLocalVectorstore in src/genkit.ts)
            await ai.index({
                indexer: menuPdfIndexer,
                documents,
            });

            return {
                success: true,
                documentsIndexed: documents.length,
            }

        } catch (err) {

            // error handling
            return {
                success: false,
                documentsIndexed: 0,
                error: err instanceof Error ? err.message : String(err),
            }
        }
    }
)

// define the menu retriever instance
export const menuRetriever = devLocalIndexerRef('menuQA');

// define the flow to question ai about the menu
export const menuQAFlow = ai.defineFlow(
    {
        name: 'menuQA',
        inputSchema: z.object({ query: z.string() }),
        outputSchema: z.object({ answer: z.string() }),
    },
    async ({ query }) => {

        // retrieve relevant documents
        const docs = await ai.retrieve({
            retriever: menuRetriever,
            query,
            options: { k: 3}, 
        });

        // generate response
        const { text } = await ai.generate({
            model: googleAI.model('gemini-2.5-flash'),
            prompt: `
                You are acting as a helpful AI assistant that can answer
                questions about the food available on the menu at Genkit Grub Pub.

                Use only the context provided to answer the question.
                If you don't know, do not make up an answer.
                Do not add or change items on the menu.

                Question: ${query}
            `,
            docs,
        });

        // return the response generated as answer
        return { answer: text };
    }
)