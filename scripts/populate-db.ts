import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { drizzle } from "drizzle-orm/postgres-js";
import postgres from "postgres";
import { env } from "@/lib/env.mjs";
import { embedMany } from "ai";
import { embeddings as embeddingsTable } from "@/lib/db/schema/embeddings";
import { openai } from "@ai-sdk/openai";
import { resources } from "@/lib/db/schema/resources";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { Document } from "langchain/document";
import fs from "fs";

const embeddingModel = openai.embedding("text-embedding-ada-002");

const client = postgres(env.DATABASE_URL);
const db = drizzle(client);

const splitter = new RecursiveCharacterTextSplitter();

const generateChunks = async (input: Document<Record<string, any>>[]) => {
  return await splitter.splitDocuments(input);
};

const generateEmbeddings = async (
  value: Document<Record<string, any>>[]
): Promise<Array<{ embedding: number[]; content: string }>> => {
  const chunks = await generateChunks(value);
  const { embeddings } = await embedMany({
    model: embeddingModel,
    values: chunks.map((c) => c.pageContent),
  });
  return embeddings.map((e, i) => ({
    content: chunks[i].pageContent,
    embedding: e,
  }));
};

const loadSampleData = async (docs: Document<Record<string, any>>[]) => {
  const content = docs.map((doc) => doc.pageContent).join("\n");
  const [resource] = await db.insert(resources).values({ content }).returning();

  const embeddings = await generateEmbeddings(docs);
  await db.insert(embeddingsTable).values(
    embeddings.map((embedding) => ({
      resourceId: resource.id,
      ...embedding,
    }))
  );
};

(async () => {
  // read all files in fintech.theodo.com
  const files = fs.readdirSync("fintech.theodo.com");
  // load sample data for each file
  for (const file of files) {
    const loader = new TextLoader(`fintech.theodo.com/${file}`);
    const docs = await loader.load();
    await loadSampleData(docs);
    console.log(`Loaded ${file}`);
  }
  console.log("Finished loading sample data");
  await client.end();
  process.exit(0);
})();
