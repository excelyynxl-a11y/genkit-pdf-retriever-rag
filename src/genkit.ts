import devLocalVectorstore, { devLocalIndexerRef } from "@genkit-ai/dev-local-vectorstore";
import googleAI from "@genkit-ai/googleai";
import { genkit } from "genkit";

export const ai = genkit({
    plugins: [
        // googleAI provides the gemini-embedding-001 embedder
        // RAG embedding technique
        googleAI(),

        // local vector store requires an embedder to translate from text to vector
        devLocalVectorstore([
            {
                indexName: 'menuQA',
                embedder: googleAI.embedder('gemini-embedding-001'),
            },
        ]),
    ]
});

export const menuPdfIndexer = devLocalIndexerRef('menuQA');