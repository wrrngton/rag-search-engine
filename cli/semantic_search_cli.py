#!/usr/bin/env python3

import argparse
import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer


def verify_model():
    instance = SemanticSearch()
    print(f"Model loaded: {instance.model}")
    print(f"Max sequence length: {instance.model.max_seq_length}")


def embed_text(text):
    instance = SemanticSearch()
    embedding = instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def embed_query_text(query):
    instance = SemanticSearch()
    embedding = instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def verify_embeddings():
    instance = SemanticSearch()
    with open(os.path.join(os.getcwd(), "data/movies.json")) as f:
        documents = json.load(f)
        embeddings = instance.load_or_create_embeddings(documents)
        print(f"Number of docs:  {len(documents)}")
        print(
            f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
        )


def chunk_text(text, chunk_size, overlap):
    chunk = text.split(" ")
    sub_arrays = [chunk[i : i + chunk_size] for i in range(0, len(chunk), chunk_size)]

    if not overlap > 0 or len(sub_arrays) == 1:
        print(f"Chunking {len(text)} characters")
        for i, sub in enumerate(sub_arrays):
            print(f"{i + 1}. {" ".join(sub)}")
        return

    overlapped_chunks = []
    num_amount_to_slice = overlap

    for i, sub in enumerate(sub_arrays):
        if i != 0:
            prev_arr = sub_arrays[i - 1]
            backwards_index = len(prev_arr) - num_amount_to_slice
            prev_el_slice = prev_arr[backwards_index:]
            joined_arrs = prev_el_slice + sub
            overlapped_chunks.append(joined_arrs)
        else:
            overlapped_chunks.append(sub)

    print(f"Chunking {len(text)} characters")
    for i, sub in enumerate(overlapped_chunks):
        print(f"{i + 1}. {" ".join(sub)}")
    return

    print(f"Chunking {len(text)} characters")
    for i, sub in enumerate(sub_arrays):
        print(f"{i + 1}. {" ".join(sub)}")


def semantic_chunk_text(text, max_chunk_size, overlap):
    semantic_chunks = re.split(r"(?<=[.!?])\s+", text)
    overlapped_chunks = []

    i = 0
    stride = max_chunk_size - overlap
    while i < len(semantic_chunks):
        print(semantic_chunks[i:i + max_chunk_size])
        i += stride


    print(f"Chunking {len(text)} characters")
    for i, sub in enumerate(overlapped_chunks):
        print(f"{i + 1}. {" ".join(sub)}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        for document in documents:
            self.document_map[document["id"]] = document
        document_string_list = [
            f"{doc['title']}: {doc['description']}" for doc in documents
        ]
        embeddings = self.model.encode(document_string_list, show_progress_bar=True)
        self.embeddings = embeddings
        np.save(os.path.join(os.getcwd(), "cache/movie_embeddings.npy"), embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        docs = documents.get("movies")
        self.documents = docs
        for document in docs:
            self.document_map[document["id"]] = document
        embeddings_path = os.path.exists(
            os.path.join(os.getcwd(), "cache/movie_embeddings.npy")
        )
        if embeddings_path:
            self.embeddings = np.load(
                os.path.join(os.getcwd(), "cache/movie_embeddings.npy")
            )
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        else:
            embeddings = self.build_embeddings(docs)
            return embeddings

    def generate_embedding(self, text):
        if not text or text == "" or text == " ":
            raise ValueError("Text must not be empty")
        embedding = self.model.encode([text])
        return embedding[0]

    def search(self, query, limit):
        # Have to do this because we haven't created a doc_id -> embedding mapping
        document_map_list = [elem for elem in self.document_map.values()]
        if not self.embeddings.any():
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embedding(query)
        embedding_doc_list = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(doc_embedding, query_embedding)
            embedding_doc_list.append((similarity, document_map_list[i]))
        embedding_doc_list.sort(key=lambda x: x[0], reverse=True)
        final_results = [
            {
                "score": d[0],
                "title": d[1].get("title"),
                "description": d[1].get("description"),
            }
            for d in embedding_doc_list[0:limit]
        ]
        for i, result in enumerate(final_results):
            print(f"""
                  {i}. {result.get('title')} (score: {result.get('score'):.4f})
                  {result.get("description")}
                  """)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    subparser.add_parser("verify", help="Very the model")

    embed_subparser = subparser.add_parser("embed_text", help="Very the model")
    embed_subparser.add_argument("text", type=str, help="Text to embed")

    verify_embeddings_subparser = subparser.add_parser(
        "verify_embeddings", help="Very embeddings"
    )

    query_embedding_subparser = subparser.add_parser(
        "embedquery", help="Generate query embeddings"
    )
    query_embedding_subparser.add_argument(
        "query", type=str, help="Query text to embed"
    )

    search_subparser = subparser.add_parser(
        "search", help="Search across embeddings with a query"
    )
    search_subparser.add_argument(
        "query", type=str, help="Query to use for embedding search"
    )
    search_subparser.add_argument(
        "--limit", type=int, default=5, nargs="*", help="Limit results"
    )

    chunk_subparser = subparser.add_parser("chunk", help="Chunk text")
    chunk_subparser.add_argument("text", type=str, help="Text to chunk")
    chunk_subparser.add_argument(
        "--chunk-size", type=int, default=200, help="Text to chunk"
    )
    chunk_subparser.add_argument(
        "--overlap", type=int, default=2, help="Text to overlap when chunking"
    )

    semantic_chunk_subparser = subparser.add_parser("semantic_chunk", help="Chunk text")
    semantic_chunk_subparser.add_argument(
        "text", type=str, help="Text to semantic_chunk"
    )
    semantic_chunk_subparser.add_argument(
        "--max-chunk-size", type=int, default=4, help="Text to semantic_chunk"
    )
    semantic_chunk_subparser.add_argument(
        "--overlap", type=int, default=0, help="Text to overlap when semantic_chunking"
    )

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "search":
            instance = SemanticSearch()
            with open(os.path.join(os.getcwd(), "data/movies.json")) as f:
                documents = json.load(f)
            instance.load_or_create_embeddings(documents)
            instance.search(args.query, limit=5)
        case "embed_text":
            embed_text(args.text)
        case "embedquery":
            embed_query_text(args.query)
        case "verify_embeddings":
            verify_embeddings()
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
