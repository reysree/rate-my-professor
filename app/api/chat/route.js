import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPrompt = `You are a helpful and knowledgeable academic assistant designed to assist students in finding the best professors according to their specific needs. When a student asks for recommendations, you will analyze their query, retrieve relevant data from your database, and generate a response with the top 3 professors that best match their criteria. Make sure to include the professor's name, subject taught, and a brief description based on reviews and ratings. Prioritize professors with higher ratings and relevant subject expertise, and ensure your recommendations are clear and informative.

Example tasks:

If a student asks for a "good physics professor," you should retrieve data on professors who teach physics and provide the top 3 based on their ratings and student reviews.
If a student asks for "professors who are great at explaining calculus," you should retrieve information on calculus professors and highlight those who excel at clear explanations.
Response Format:

Professor Name: [Name]
Subject: [Subject]
Rating: [Star Rating]
Review Summary: [Brief review summary]
Always remember to be polite, concise, and supportive in your responses, helping students make the best choice based on your recommendations.`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });

  const index = pc.index("rag").namespace("ns1");
  const openai = new OpenAI();

  const text = data[data.length - 1].content;
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString =
    "\n\nReturned Results from vector db(done automatically): ";
  results.matches.forEach((match) => {
    resultString += `\n
    
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n
    `;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPrompt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-4o-mini",
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
