import express from 'express';
import { createClient } from '@supabase/supabase-js';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { SupabaseVectorStore } from 'langchain/vectorstores/supabase';
import { compile } from 'html-to-text';
import { RecursiveUrlLoader } from 'langchain/document_loaders/web/recursive_url';
import { PromptTemplate } from 'langchain/prompts';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { StringOutputParser } from 'langchain/schema/output_parser';
import { RunnablePassthrough, RunnableSequence } from 'langchain/schema/runnable';
import 'dotenv/config';

const app = express();
const port = process.env.PORT || 3000;
app.use(express.json());

// Existing conversation history storage
const userConvHistories = {};

// Function to format conversation history messages
const formatConvHistory = (messages) => messages.join('\n');

// Supabase configuration
const sbUrl = process.env.SB_URL;
const sbApiKey = process.env.SBAPI_KEY;
const openAIApiKey = process.env.OPENAI_API_KEY;

// Initialize Supabase client
const client = createClient(sbUrl, sbApiKey);

// Initialize OpenAI embeddings, vector store, and retriever
const embeddings = new OpenAIEmbeddings({ openAIApiKey });
const vectorStore = new SupabaseVectorStore(embeddings, {
  client,
  tableName: 'documents',
  queryName: 'match_documents',
});
const retriever = vectorStore.asRetriever();

// Initialize ChatOpenAI instance
const llm = new ChatOpenAI({ openAIApiKey });

// Templates for standalone questions and answers
const standAloneQuestionTemplate = `Given a conversation history (if any) and a question, convert it to a standalone question.\nConversation history: {conv_history}\nQuestion: {question} Standalone question:`;

const answerTemplate = `You are a helpful and enthusiastic support bot who can answer a given question about Scrimba based on the context provided and the conversation history. Try to find the answer in the context. If the answer is not given in the context, find the answer in the conversation history if possible. If you really don't know the answer, say "I'm sorry, I don't know the answer to that." And direct the questioner to email help@company.com. Don't try to make up an answer. Always speak as if you were chatting to a friend.\nContext: {context}\nConversation history: {conv_history}\nQuestion: {question}\nAnswer: `;

// Initialize prompt templates and chains
const answerPrompt = PromptTemplate.fromTemplate(answerTemplate);
const retrieverChain = RunnableSequence.from([
  (prevResult) => prevResult.standalone_question,
  retriever,
  combineDocuments,
]);
const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());
const standaloneQuestionPrompt = PromptTemplate.fromTemplate(
  standAloneQuestionTemplate
);
const standaloneQuestionChain = standaloneQuestionPrompt
  .pipe(llm)
  .pipe(new StringOutputParser());

// Main chain for processing questions and generating responses
const chain = RunnableSequence.from([
  {
    standalone_question: standaloneQuestionChain,
    original_input: new RunnablePassthrough(),
  },
  {
    context: retrieverChain,
    question: ({ original_input }) => original_input.question,
    conv_history: ({ original_input }) => original_input.conv_history,
  },
  answerChain,
]);

// Route for handling chat requests
app.post('/api/chat', async (req, res) => {
  try {
    const { question } = req.body;
    const userId = req.ip;

    // Get or initialize conversation history for the user
    const convHistory = userConvHistories[userId] || [];

    // Append the new question to the conversation history
    convHistory.push(`Human: ${question}`);

    // Process the question and generate a response
    const response = await chain.invoke({
      question,
      conv_history: formatConvHistory(convHistory),
    });

    // Append the AI response to the conversation history
    convHistory.push(`AI: ${response}`);

    // Save the updated conversation history for the user
    userConvHistories[userId] = convHistory;

    // Respond with the generated response
    res.json({ response });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// New API endpoint for loading documents from a URL
app.post('/api/load-documents', async (req, res) => {
  try {
    const url = req.body.url;

    // Initialize HTML-to-text converter
    const compiledConvert = compile({ wordwrap: 1000 });

    // Initialize RecursiveUrlLoader
    const loader = new RecursiveUrlLoader(url, {
      extractor: compiledConvert,
      maxDepth: 1000,
    });

    // Load documents from the URL
    const docs = await loader.load();

    // Initialize RecursiveCharacterTextSplitter
    const splitter = new RecursiveCharacterTextSplitter();

    // Split the documents into characters
    const output = await splitter.createDocuments([JSON.stringify(docs)]);

    // Store the documents in the Supabase vector store
    await SupabaseVectorStore.fromDocuments(output, embeddings, {
      client,
      tableName: 'documents',
    });

    res.json({ success: true });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

// Function to combine document pages
function combineDocuments(docs) {
  console.log(docs);
  return docs.map((doc) => doc.pageContent).join('\n\n');
}

// Start the Express server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
