import argparse
import json
import os
import string
import pickle

from nltk.stem import PorterStemmer

MOVIES_DATA = os.path.join(os.getcwd(), "data/movies.json")
STOP_WORDS_DATA = os.path.join(os.getcwd(), "data/stopwords.txt")

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap= {}

    def __add_document(self, doc_id, text):
        tokens = normalise(text)

        for token in tokens:
            if not self.index.get(token):
                self.index[token] = []
                self.index[token].append(doc_id)
            else:
                self.index[token].append(doc_id)

    def build(self):
        with open(MOVIES_DATA, "r") as f:
            json_data = json.load(f)
            movies = json_data.get("movies")

            if not movies:
                print("no data loaded")
                return

            for movie in movies:
                self.__add_document(
                    movie.get(
                        "id"), f"{movie.get('title')} {movie.get('description')}"
                )
                self.docmap[movie.get("id")] = movie

    def get_documents(self, term):
        return self.index.get(term.lower())

    def get_index(self):
        return self.docmap

    def save(self):
        cache_path = os.path.join(os.getcwd(), "cache")

        if not os.path.exists(cache_path):
            os.mkdir(os.path.join(os.getcwd, "cache"))

        with open(os.path.join(cache_path, "index.pkl"), "w+") as f:
            pickle.dump(self.index, f)

        with open(os.path.join(cache_path, "docmap.pkl"), "w+") as f:
            pickle.dump(self.docmap, f)


with open(STOP_WORDS_DATA, "r") as sw:
    stop_words = [sw.replace("\n", "") for sw in sw.readlines()]


def normalise(input_term: str) -> list:
    # Remove punctuations
    punc_table = str.maketrans("", "", string.punctuation)
    translated_term = input_term.translate(punc_table)
    new_term = translated_term.lower()

    # Tokenize
    split_term = new_term.split(" ")
    split_tokens = [term for term in split_term if term != ""]

    # Remove stop words
    stop_word_stripped_tokens = [
        term for term in split_tokens if term not in stop_words
    ]

    # Stemm
    stemmed_tokens = [stemmer.stem(term) for term in stop_word_stripped_tokens]

    return stemmed_tokens


def search(query: str) -> list:
    results = []
    with open(MOVIES_DATA, "r") as f:
        movies = json.load(f)
        normalised_query = normalise(query)

        for m in movies.get("movies"):
            normalised_title = normalise(m.get("title"))

            for q in normalised_query:
                for t in normalised_title:
                    if q in t and m not in results:
                        results.append(m)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")
    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser = subparsers.add_parser(
        "build", help="Build an inverted index")

    args = parser.parse_args()

    match args.command:
        case "build":
            indexer = InvertedIndex()
            indexer.build()
            # blah = indexer.get_index()
            indexer.save()
        case "search":
            search_results = search(args.query)
            f_str = f"Searching for {args.query}\n"
            for i, m in enumerate(search_results[0:5]):
                f_str += f"{i + 1}. {m.get('title')}\n"
            print(f_str)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
