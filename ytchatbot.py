from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Document ingestion
video_id = "Gfr50f6ZBvo"  # only the ID, not full URL (compulsory bro)

try:
    # Create API instance
    api = YouTubeTranscriptApi()
    # Fetch transcript
    transcript_list = api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    # transcript = " ".join(chunk["text"] for chunk in transcript_list)
    transcript = " ".join(getattr(chunk, "text", "") for chunk in transcript_list)

    # print("Transcript fetched successfully ✅\n")
    # print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

#step 1b - Text Splitting 

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])


#step 1c & 1d -- Embedding generaton and storage of embeddings in vector store (faiss)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)  #faiss vector store

print(vector_store.get_by_ids(['2dbc267f-db52-432f-aab2-e7610ea7c345']))


#step 1 creation of external knowledge source is created

#step 2  - retrieval

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(retriever.invoke('What is deepmind'))