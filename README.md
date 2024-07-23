RAG Chatbot Guide
In this guide, you will learn how to build a retrieval-augmented generation (RAG) chatbot application.

Before we dive in, let's look at what RAG is, and why we would want to use it.

What is RAG?
RAG stands for retrieval augmented generation. In simple terms, RAG is the process of providing a Large Language Model (LLM) with specific information relevant to the prompt.

Why is RAG important?
While LLMs are powerful, the information they can reason on is restricted to the data they were trained on. This problem becomes apparent when asking an LLM for information outside of their training data, like proprietary data or common knowledge that has occurred after the model’s training cutoff. RAG solves this problem by fetching information relevant to the prompt and then passing that to the model as context.

To illustrate with a basic example, imagine asking the model for your favorite food:

**input**
What is my favorite food?

**generation**
I don't have access to personal information about individuals, including their
favorite foods.
Not surprisingly, the model doesn’t know. But imagine, alongside your prompt, the model received some extra context:

**input**
Respond to the user's prompt using only the provided context.
user prompt: 'What is my favorite food?'
context: user loves chicken nuggets

**generation**
Your favorite food is chicken nuggets!
Just like that, you have augmented the model’s generation by providing relevant information to the query. Assuming the model has the appropriate information, it is now highly likely to return an accurate response to the users query. But how does it retrieve the relevant information? The answer relies on a concept called embedding.

You could fetch any context for your RAG application (eg. Google search). Embeddings and Vector Databases are just a specific retrieval approach to achieve semantic search.

Embedding
Embeddings are a way to represent words, phrases, or images as vectors in a high-dimensional space. In this space, similar words are close to each other, and the distance between words can be used to measure their similarity.

In practice, this means that if you embedded the words cat and dog, you would expect them to be plotted close to each other in vector space. The process of calculating the similarity between two vectors is called ‘cosine similarity’ where a value of 1 would indicate high similarity and a value of -1 would indicate high opposition.

Don’t worry if this seems complicated. a high level understanding is all you need to get started! For a more in-depth introduction to embeddings, check out this guide.

As mentioned above, embeddings are a way to represent the semantic meaning of words and phrases. The implication here is that the larger the input to your embedding, the lower quality the embedding will be. So how would you approach embedding content longer than a simple phrase?

Chunking
Chunking refers to the process of breaking down a particular source material into smaller pieces. There are many different approaches to chunking and it’s worth experimenting as the most effective approach can differ by use case. A simple and common approach to chunking (and what you will be using in this guide) is separating written content by sentences.

Once your source material is appropriately chunked, you can embed each one and then store the embedding and the chunk together in a database. Embeddings can be stored in any database that supports vectors. For this tutorial, you will be using Postgres alongside the pgvector plugin.

RAG Guide 1

All Together Now
Combining all of this together, RAG is the process of enabling the model to respond with information outside of it’s training data by embedding a users query, retrieving the relevant source material (chunks) with the highest semantic similarity, and then passing them alongside the initial query as context. Going back to the example where you ask the model for your favorite food, the prompt preparation process would look like this.

RAG Guide 2

By passing the appropriate context and refining the model’s objective, you are able to fully leverage its strengths as a reasoning machine.

Onto the project!

Project Setup
In this project, you will build a chatbot that will only respond with information that it has within its knowledge base. The chatbot will be able to both store and retrieve information. This project has many interesting use cases from customer support through to building your own second brain!

This project will use the following stack:

Next.js 14 (App Router)
Vercel AI SDK
OpenAI
Drizzle ORM
Postgres with pgvector
shadcn-ui and TailwindCSS for styling
Clone Repo
To reduce the scope of this guide, you will be starting with a repository that already has a few things set up for you:

Drizzle ORM (lib/db) including an initial migration and a script to migrate (db:migrate)
a basic schema for the resources table (this will be for source material)
a Server Action for creating a resource
To get started, clone the starter repository with the following command:

git clone https://github.com/vercel/ai-sdk-rag-starter
cd ai-sdk-rag-starter
First things first, run the following command to install the project’s dependencies:

pnpm install
Create Database
You will need a Postgres database to complete this tutorial. If you don’t have Postgres setup on your local machine you can:

Create a free Postgres database with Vercel Postgres; or
Follow this guide to set it up locally
Migrate Database
Once you have a Postgres database, you need to add the connection string as an environment secret.

Make a copy of the .env.example file and rename it to .env.

cp .env.example .env
Open the new .env file. You should see an item called DATABASE_URL. Copy in your database connection string after the equals sign.

With that set up, you can now run your first database migration. Run the following command:

pnpm db:migrate
This will first add the pgvector extension to your database. Then it will create a new table for your resources schema that is defined in lib/db/schema/resources.ts. This schema has four columns: id, content, createdAt, and updatedAt.

If you experience an error with the migration, open your migration file (lib/db/migrations/0000_yielding_bloodaxe.sql), cut (copy and remove) the first line, and run it directly on your postgres instance. You should now be able to run the updated migration. More info.

OpenAI API Key
For this guide, you will need an OpenAI API key. To generate an API key, go to platform.openai.com.

Once you have your API key, paste it into your .env file (OPENAI_API_KEY).

Build
Let’s build a quick task list of what needs to be done:

Create a table in your database to store embeddings
Add logic to chunk and create embeddings when creating resources
Create a chatbot
Give the chatbot tools to query / create resources for it’s knowledge base
Create Embeddings Table
Currently, your application has one table (resources) which has a column (content) for storing content. Remember, each resource (source material) will have to be chunked, embedded, and then stored. Let’s create a table called embeddings to store these chunks.

Create a new file (lib/db/schema/embeddings.ts) and add the following code:

lib/db/schema/embeddings.ts

import { nanoid } from '@/lib/utils';
import { index, pgTable, text, varchar, vector } from 'drizzle-orm/pg-core';
import { resources } from './resources';

export const embeddings = pgTable(
'embeddings',
{
id: varchar('id', { length: 191 })
.primaryKey()
.$defaultFn(() => nanoid()),
resourceId: varchar('resource_id', { length: 191 }).references(
() => resources.id,
{ onDelete: 'cascade' },
),
content: text('content').notNull(),
embedding: vector('embedding', { dimensions: 1536 }).notNull(),
},
table => ({
embeddingIndex: index('embeddingIndex').using(
'hnsw',
table.embedding.op('vector_cosine_ops'),
),
}),
);
This table has four columns:

id - unique identifier
resourceId - a foreign key relation to the full source material
content - the plain text chunk
embedding - the vector representation of the plain text chunk
To perform similarity search, you also need to include an index (HNSW or IVFFlat) on this column for better performance.

To push this change to the database, run the following command:

pnpm db:push
Add Embedding Logic
Now that you have a table to store embeddings, it’s time to write the logic to create the embeddings.

Create a file with the following command:

mkdir lib/ai && touch lib/ai/embedding.ts
Generate Chunks
Remember, to create an embedding, you will start with a piece of source material (unknown length), break it down into smaller chunks, embed each chunk, and then save the chunk to the database. Let’s start by creating a function to break the source material into small chunks.

lib/ai/embedding.ts

const generateChunks = (input: string): string[] => {
return input
.trim()
.split('.')
.filter(i => i !== '');
};
This function will take an input string and split it by periods, filtering out any empty items. This will return an array of strings. It is worth experimenting with different chunking techniques in your projects as the best technique will vary.

Install AI SDK
You will use the Vercel AI SDK to create embeddings. This will require two more dependencies, which you can install by running the following command:

pnpm install ai @ai-sdk/openai
This will install the Vercel AI SDK and the OpenAI provider.

Vercel AI SDK is designed to be a unified interface to interact with any large language model. This means that you can change model and providers with just one line of code! Learn more about available providers and building custom providers in the providers section.

Generate Embeddings
Let’s add a function to generate embeddings. Copy the following code into your lib/ai/embedding.ts file.

lib/ai/embedding.ts

import { embedMany } from 'ai';
import { openai } from '@ai-sdk/openai';

const embeddingModel = openai.embedding('text-embedding-ada-002');

const generateChunks = (input: string): string[] => {
return input
.trim()
.split('.')
.filter(i => i !== '');
};

export const generateEmbeddings = async (
value: string,
): Promise<Array<{ embedding: number[]; content: string }>> => {
const chunks = generateChunks(value);
const { embeddings } = await embedMany({
model: embeddingModel,
values: chunks,
});
return embeddings.map((e, i) => ({ content: chunks[i], embedding: e }));
};
In this code, you first define the model you want to use for the embeddings. In this example, you are using OpenAI’s text-embedding-ada-002 embedding model.

Next, you create an asynchronous function called generateEmbeddings. This function will take in the source material (value) as an input and return a promise of an array of objects, each containing an embedding and content. Within the function, you first generate chunks for the input. Then, you pass those chunks to the embedMany function imported from the Vercel AI SDK which will return embeddings of the chunks you passed in. Finally, you map over and return the embeddings in a format that is ready to save in the database.

Update Server Action
Open the file at lib/actions/resources.ts. This file has one function, createResource, which, as the name implies, allows you to create a resource.

lib/actions/resources.ts

'use server';

import {
NewResourceParams,
insertResourceSchema,
resources,
} from '@/lib/db/schema/resources';
import { db } from '../db';

export const createResource = async (input: NewResourceParams) => {
try {
const { content } = insertResourceSchema.parse(input);

    const [resource] = await db
      .insert(resources)
      .values({ content })
      .returning();

    return 'Resource successfully created.';

} catch (e) {
if (e instanceof Error)
return e.message.length > 0 ? e.message : 'Error, please try again.';
}
};
This function is a Server Action, as denoted by the “use server”; directive at the top of the file. This means that it can be called anywhere in your Next.js application. This function will take an input, run it through a Zod schema to ensure it adheres to the correct schema, and then creates a new resource in the database. This is the ideal location to generate and store embeddings of the newly created resources.

Update the file with the following code:

lib/actions/resources.ts

'use server';

import {
NewResourceParams,
insertResourceSchema,
resources,
} from '@/lib/db/schema/resources';
import { db } from '../db';
import { generateEmbeddings } from '../ai/embedding';
import { embeddings as embeddingsTable } from '../db/schema/embeddings';

export const createResource = async (input: NewResourceParams) => {
try {
const { content } = insertResourceSchema.parse(input);

    const [resource] = await db
      .insert(resources)
      .values({ content })
      .returning();

    const embeddings = await generateEmbeddings(content);
    await db.insert(embeddingsTable).values(
      embeddings.map(embedding => ({
        resourceId: resource.id,
        ...embedding,
      })),
    );

    return 'Resource successfully created and embedded.';

} catch (error) {
return error instanceof Error && error.message.length > 0
? error.message
: 'Error, please try again.';
}
};
First, you call the generateEmbeddings function created in the previous step, passing in the source material (content). Once you have your embeddings (e) of the source material, you can save them to the database, passing the resourceId alongside each embedding.

Create Root Page
Great! Let's build the frontend. Vercel AI SDK’s useChat hook allows you to easily create a conversational user interface for your chatbot application.

Replace your root page (app/page.tsx) with the following code.

app/page.tsx

'use client';

import { useChat } from 'ai/react';

export default function Chat() {
const { messages, input, handleInputChange, handleSubmit } = useChat();
return (
<div className="flex flex-col w-full max-w-md py-24 mx-auto stretch">
<div className="space-y-4">
{messages.map(m => (
<div key={m.id} className="whitespace-pre-wrap">
<div>
<div className="font-bold">{m.role}</div>
<p>{m.content}</p>
</div>
</div>
))}
</div>

      <form onSubmit={handleSubmit}>
        <input
          className="fixed bottom-0 w-full max-w-md p-2 mb-8 border border-gray-300 rounded shadow-xl"
          value={input}
          placeholder="Say something..."
          onChange={handleInputChange}
        />
      </form>
    </div>

);
}
The useChat hook enables the streaming of chat messages from your AI provider (you will be using OpenAI), manages the state for chat input, and updates the UI automatically as new messages are received.

Run the following command to start the Next.js dev server:

pnpm run dev
Head to http://localhost:3000. You should see an empty screen with an input bar floating at the bottom. Try to send a message. The message shows up in the UI for a fraction of a second and then disappears. This is because you haven’t set up the corresponding API route to call the model! By default, useChat will send a POST request to the /api/chat endpoint with the messages as the request body.

You can customize the endpoint in the useChat configuration object
Create API Route
In Next.js, you can create custom request handlers for a given route using Route Handlers. Route Handlers are defined in a route.ts file and can export HTTP methods like GET, POST, PUT, PATCH etc.

Create a file at app/api/chat/route.ts by running the following command:

mkdir -p app/api/chat && touch app/api/chat/route.ts
Open the file and add the following code:

app/api/chat/route.ts

import { openai } from '@ai-sdk/openai';
import { convertToCoreMessages, streamText } from 'ai';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
const { messages } = await req.json();

const result = await streamText({
model: openai('gpt-4o'),
messages: convertToCoreMessages(messages),
});

return result.toAIStreamResponse();
}
In this code, you declare and export an asynchronous function called POST. You retrieve the messages from the request body and then pass them to the streamText function imported from the Vercel AI SDK, alongside the model you would like to use. Finally, you return the model’s response in AIStreamResponse format.

Head back to the browser and try to send a message again. You should see a response from the model streamed directly in!

Refining your prompt
While you now have a working chatbot, it isn't doing anything special.

Let’s add system instructions to refine and restrict the model’s behavior. In this case, you want the model to only use information it has retrieved to generate responses. Update your route handler with the following code:

app/api/chat/route.ts

import { openai } from '@ai-sdk/openai';
import { convertToCoreMessages, streamText } from 'ai';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
const { messages } = await req.json();

const result = await streamText({
model: openai('gpt-4o'),
system: `You are a helpful assistant. Check your knowledge base before answering any questions.
    Only respond to questions using information from tool calls.
    if no relevant information is found in the tool calls, respond, "Sorry, I don't know."`,
messages: convertToCoreMessages(messages),
});

return result.toAIStreamResponse();
}
Head back to the browser and try to ask the model what your favorite food is. The model should now respond exactly as you instructed above (“Sorry, I don’t know”) given it doesn’t have any relevant information.

In its current form, your chatbot is now, well, useless. How do you give the model the ability to add and query information?

Using Tools
A tool is a function that can be called by the model to perform a specific task. You can think of a tool like a program you give to the model that it can run as and when it deems necessary.

Let’s see how you can create a tool to give the model the ability to create, embed and save a resource to your chatbots’ knowledge base.

Add Resource Tool
Update your route handler with the following code:

app/api/chat/route.ts

import { createResource } from '@/lib/actions/resources';
import { openai } from '@ai-sdk/openai';
import { convertToCoreMessages, streamText, tool } from 'ai';
import { z } from 'zod';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
const { messages } = await req.json();

const result = await streamText({
model: openai('gpt-4o'),
system: `You are a helpful assistant. Check your knowledge base before answering any questions.
    Only respond to questions using information from tool calls.
    if no relevant information is found in the tool calls, respond, "Sorry, I don't know."`,
messages: convertToCoreMessages(messages),
tools: {
addResource: tool({
description: `add a resource to your knowledge base.
          If the user provides a random piece of knowledge unprompted, use this tool without asking for confirmation.`,
parameters: z.object({
content: z
.string()
.describe('the content or resource to add to the knowledge base'),
}),
execute: async ({ content }) => createResource({ content }),
}),
},
});

return result.toAIStreamResponse();
}
In this code, you define a tool called addResource. This tool has three elements:

description: description of the tool that will influence when the tool is picked.
parameters: Zod schema that defines the parameters necessary for the tool to run.
execute: An asynchronous function that is called with the arguments from the tool call.
In simple terms, on each generation, the model will decide whether it should call the tool. If it deems it should call the tool, it will extract the parameters from the input and then append a new message to the messages array of type tool-call. The AI SDK will then run the execute function with the parameters provided by the tool-call message.

Head back to the browser and tell the model your favorite food. You should see an empty response in the UI. Did anything happen? Let’s see. Run the following command in a new terminal window.

pnpm db:studio
This will start Drizzle Studio where we can view the rows in our database. You should see a new row in both the embeddings and resources table with your favorite food!

Let’s make a few changes in the UI to communicate to the user when a tool has been called. Head back to your root page (app/page.tsx) and add the following code:

app/page.tsx

'use client';

import { useChat } from 'ai/react';

export default function Chat() {
const { messages, input, handleInputChange, handleSubmit } = useChat();
return (
<div className="flex flex-col w-full max-w-md py-24 mx-auto stretch">
<div className="space-y-4">
{messages.map(m => (
<div key={m.id} className="whitespace-pre-wrap">
<div>
<div className="font-bold">{m.role}</div>
<p>
{m.content.length > 0 ? (
m.content
) : (
<span className="italic font-light">
{'calling tool: ' + m?.toolInvocations?.[0].toolName}
</span>
)}
</p>
</div>
</div>
))}
</div>

      <form onSubmit={handleSubmit}>
        <input
          className="fixed bottom-0 w-full max-w-md p-2 mb-8 border border-gray-300 rounded shadow-xl"
          value={input}
          placeholder="Say something..."
          onChange={handleInputChange}
        />
      </form>
    </div>

);
}
With this change, you now conditionally render the tool that has been called directly in the UI. Save the file and head back to browser. Tell the model your favorite movie. You should see which tool is called in place of the model’s typical text response.

Improving UX with Tool Roundtrips
It would be nice if the model could summarize the action too. However, technically, once the model calls a tool, it has completed its generation as it ‘generated’ a tool call. How could you achieve this desired behaviour?

The AI SDK has a feature called maxToolCallRoundtrips which will automatically send tool call results back to the model!

Open your root page (app/page.tsx) and add the following key to the useChat configuration object:

app/page.tsx

// ... Rest of your code

const { messages, input, handleInputChange, handleSubmit } = useChat({
maxToolRoundtrips: 2,
});

// ... Rest of your code
Head back to the browser and tell the model your favorite pizza topping (note: pineapple is not an option). You should see a follow-up response from the model confirming the action.

Retrieve Resource Tool
The model can now add and embed arbitrary information to your knowledge base. However, it still isn’t able to query it. Let’s create a new tool to allow the model to answer questions by finding relevant information in your knowledge base.

To find similar content, you will need to embed the users query, search the database for semantic similarities, then pass those items to the model as context alongside the query. To achieve this, let’s update your embedding logic file (lib/ai/embedding.ts):

lib/ai/embedding.ts

import { embed, embedMany } from 'ai';
import { openai } from '@ai-sdk/openai';
import { db } from '../db';
import { cosineDistance, desc, gt, sql } from 'drizzle-orm';
import { embeddings } from '../db/schema/embeddings';

const embeddingModel = openai.embedding('text-embedding-ada-002');

const generateChunks = (input: string): string[] => {
return input
.trim()
.split('.')
.filter(i => i !== '');
};

export const generateEmbeddings = async (
value: string,
): Promise<Array<{ embedding: number[]; content: string }>> => {
const chunks = generateChunks(value);
const { embeddings } = await embedMany({
model: embeddingModel,
values: chunks,
});
return embeddings.map((e, i) => ({ content: chunks[i], embedding: e }));
};

export const generateEmbedding = async (value: string): Promise<number[]> => {
const input = value.replaceAll('\\n', ' ');
const { embedding } = await embed({
model: embeddingModel,
value: input,
});
return embedding;
};

export const findRelevantContent = async (userQuery: string) => {
const userQueryEmbedded = await generateEmbedding(userQuery);
const similarity = sql<number>`1 - (${cosineDistance(
    embeddings.embedding,
    userQueryEmbedded,
  )})`;
const similarGuides = await db
.select({ name: embeddings.content, similarity })
.from(embeddings)
.where(gt(similarity, 0.5))
.orderBy(t => desc(t.similarity))
.limit(4);
return similarGuides;
};
In this code, you add two functions:

generateEmbedding: generate a single embedding from an input string
findRelevantContent: embeds the user’s query, searches the database for similar items, then returns relevant items
With that done, it’s onto the final step: creating the tool.

Go back to your route handler (api/chat/route.ts) and add a new tool called getInformation:

api/chat/route.ts

import { createResource } from '@/lib/actions/resources';
import { openai } from '@ai-sdk/openai';
import { convertToCoreMessages, streamText, tool } from 'ai';
import { z } from 'zod';
import { findRelevantContent } from '@/lib/ai/embedding';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export async function POST(req: Request) {
const { messages } = await req.json();

const result = await streamText({
model: openai('gpt-4o'),
messages: convertToCoreMessages(messages),
system: `You are a helpful assistant. Check your knowledge base before answering any questions.
    Only respond to questions using information from tool calls.
    if no relevant information is found in the tool calls, respond, "Sorry, I don't know."`,
tools: {
addResource: tool({
description: `add a resource to your knowledge base.
          If the user provides a random piece of knowledge unprompted, use this tool without asking for confirmation.`,
parameters: z.object({
content: z
.string()
.describe('the content or resource to add to the knowledge base'),
}),
execute: async ({ content }) => createResource({ content }),
}),
getInformation: tool({
description: `get information from your knowledge base to answer questions.`,
parameters: z.object({
question: z.string().describe('the users question'),
}),
execute: async ({ question }) => findRelevantContent(question),
}),
},
});

return result.toAIStreamResponse();
}
Head back to the browser, refresh the page, and ask for your favorite food. You should see the model call the getInformation tool, and then use the relevant information to formulate a response!

Conclusion
Congratulations, you have successfully built an AI chatbot that can dynamically add and retrieve information to and from a knowledge base. Throughout this guide, you learned how to create and store embeddings, set up server actions to manage resources, and use tools to extend the capabilities of your chatbot.
