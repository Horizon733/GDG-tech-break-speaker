from typing import List
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, UnstructuredFileLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings


def extract_documents(docs_folder: str) -> List[Document]:
    print(f"Extracting files from: {Path(docs_folder).absolute()}")
    if not Path(docs_folder).exists():
        raise SystemExit(f"Directory '{docs_folder}' does not exist.")

    pdf_loader = DirectoryLoader(
        docs_folder,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )

    other_loader = DirectoryLoader(
        docs_folder,
        glob="**/*.txt",
        loader_cls=UnstructuredFileLoader,
        show_progress=True,
    )

    documents = pdf_loader.load() + other_loader.load() 
    return documents

def create_chunks(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    # I have been teaching RAG since a year -> [{"id": "", chunks: ["I", "h", "a"....]}, {"id": []}]
    return text_splitter.split_documents(documents)

def embeddings_factory() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def calculate_chunk_id(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_chroma(chunks) -> bool:
    db = Chroma(
        persist_directory="chroma",
        embedding_function=embeddings_factory()
    )
    chunk_ids = calculate_chunk_id(chunks)

    existing_item = db.get(include=[])
    existing_ids = set(existing_item["ids"])
    print(existing_ids)

    new_chunks = []
    for chunk in chunk_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
        return True
    else:
        print("✅ No new documents to add")
        return False


def main():
    docs_folder = "docs"
    chunk_size = 800
    chunk_overlap = 80
# [800] -> 800 character
# [80] -> 80 character overlap
    docs = extract_documents(docs_folder)
    print(f"{len(docs)} documents extracted.")
    chunks = create_chunks(docs, chunk_size, chunk_overlap)
    is_added = add_to_chroma(chunks)
    if is_added:
        print("Chroma collection created successfully")
    else:
        print("Not added")

if __name__ == "__main__":
    main()
