#!/usr/bin/env python3

import argparse

from sentence_transformers import SentenceTransformer


def verify_model():
    instance = SemanticSearch()
    print(f"Model loaded: {instance.model}")
    print(f"Max sequence length: {instance.model.max_seq_length}")


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def add_vectors(self, vec1, vec2):
        if len(vec1) != len(vec2):
            raise ValueError("Vectors aren't the same length")
        new_vec = []
        for i, val in enumerate(vec1):
            new_vec.append(val + vec2[i])
        print(new_vec)


def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")
    subparser.add_parser("verify", help="Very the model")

    vec_subparser = subparser.add_parser("addvecs", help="Very the model")
    vec_subparser.add_argument("vec1", type=int, nargs="+", help="vec1")
    vec_subparser.add_argument("vec2", type=int, nargs="+", help="vec2")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "addvecs":
            print(type(args.vec1[0]))
            index = SemanticSearch()
            index.add_vectors(args.vec1, args.vec2)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
