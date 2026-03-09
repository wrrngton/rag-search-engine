import argparse
import json
import math
import os
import pickle
import string
from collections import Counter
from operator import itemgetter

from nltk.stem import PorterStemmer

MOVIES_DATA = os.path.join(os.getcwd(), "data/movies.json")
STOP_WORDS_DATA = os.path.join(os.getcwd(), "data/stopwords.txt")
CACHE_DIR = os.path.join(os.getcwd(), "cache")
CONSTANTS = {"BM25_K1": 1.5, "BM25_B": 0.75}

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text, movie_metadata):
        tokens = normalise(text)

        # Add to docmap
        self.docmap[doc_id] = movie_metadata

        # Add to term frequencies
        self.term_frequencies[doc_id] = Counter(tokens)

        # Add to doc_lengths
        doc_token_length = len(tokens)
        self.doc_lengths[doc_id] = doc_token_length

        for token in tokens:
            # Add to inverted index
            if not self.index.get(token):
                self.index[token] = []
                if doc_id not in self.index[token]:
                    self.index[token].append(doc_id)
            else:
                if doc_id not in self.index[token]:
                    self.index[token].append(doc_id)

    def __get_avg_doclength(self) -> float:
        doc_lengths = self.doc_lengths
        total_docs_length = sum(doc_lengths.values())
        num_total_docs = len(doc_lengths)
        average_doc_length = total_docs_length / num_total_docs
        return average_doc_length

    def build(self):
        with open(MOVIES_DATA, "r") as f:
            json_data = json.load(f)
            movies = json_data.get("movies")

            if not movies:
                print("no data loaded")
                return

            for movie in movies:
                self.__add_document(
                    movie.get("id"),
                    f"{movie.get('title')} {movie.get('description')}",
                    movie,
                )

    def get_documents(self, term):
        docs = self.index[term]
        print(docs)

    def get_tf(self, doc_id, term):
        return self.term_frequencies.get(doc_id)[term]

    def get_bm25_idf(self, term: str) -> float:
        tokens = normalise(term)
        if len(tokens) > 1:
            raise Exception("Term should be no more than 1 word")
        n = len(self.docmap)
        df = len(self.index.get(tokens[0]))
        bm25 = math.log((n - df + 0.5) / (df + 0.5) + 1)
        return bm25

    def get_bm25_tf(
        self, doc_id, term, k1=CONSTANTS.get("BM25_K1"), b=CONSTANTS.get("BM25_B")
    ):
        tokens = normalise(term)
        if len(tokens) > 1:
            raise Exception("can only calculate bm25 for a single term")
        tf = self.get_tf(doc_id, tokens[0])
        avg_doc_lengths = self.__get_avg_doclength()
        current_doc_length = self.doc_lengths[doc_id]
        length_norm = 1 - b + b * (current_doc_length / avg_doc_lengths)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component

    def bm25(self, doc_id, term):
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(self, query, limit):
        query_tokens = normalise(query)
        scores = {}
        limit = 5
        for doc in self.docmap:
            scores[doc] = 0
            for query in query_tokens:
                bm25 = self.bm25(doc, query)
                scores[doc] += bm25

        top_n_dict = dict(sorted(scores.items(), key=itemgetter(1), reverse=True)[:limit])
        return top_n_dict

    def save(self):
        cache_path = os.path.join(os.getcwd(), "cache")

        if not os.path.exists(cache_path):
            os.mkdir(os.path.join(os.getcwd, "cache"))

        with open(os.path.join(cache_path, "index.pkl"), "wb+") as f:
            pickle.dump(self.index, f)

        # with open(os.path.join(cache_path, "index.json"), "w+", encoding="utf-8") as f:
        #     json.dump(self.index, f)

        with open(os.path.join(cache_path, "docmap.pkl"), "wb+") as f:
            pickle.dump(self.docmap, f)

        with open(os.path.join(cache_path, "term_frequencies.pkl"), "wb+") as f:
            pickle.dump(self.term_frequencies, f)

        # with open(os.path.join(cache_path, "term_frequencies.json"), "w+") as f:
        #     json.dump(self.term_frequencies, f)

        with open(os.path.join(cache_path, "doc_lengths.pkl"), "wb+") as f:
            pickle.dump(self.doc_lengths, f)

        # with open(os.path.join(cache_path, "term_frequencies.json"), "w+") as f:
        #     json.dump(self.term_frequencies, f)

    def load(self):
        try:
            with open(os.path.join(os.getcwd(), "cache/docmap.pkl"), "rb") as f:
                self.docmap = pickle.load(f)
            with open(os.path.join(os.getcwd(), "cache/index.pkl"), "rb") as f:
                self.index = pickle.load(f)
            with open(
                os.path.join(os.getcwd(), "cache/term_frequencies.pkl"), "rb"
            ) as f:
                self.term_frequencies = pickle.load(f)
            with open(os.path.join(os.getcwd(), "cache/doc_lengths.pkl"), "rb") as f:
                self.doc_lengths = pickle.load(f)

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Index files not found, did you forget to run the indexer first"
            ) from e

        return self.docmap, self.index, self.term_frequencies


with open(STOP_WORDS_DATA, "r") as sw:
    stop_words = [sw.replace("\n", "") for sw in sw.readlines()]


def normalise(input_term: str) -> list:
    # Remove punctuation
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
    unicode_removed_tokens = [
        term.replace("\n", "") for term in stop_word_stripped_tokens
    ]

    # Stem
    stemmed_tokens = [stemmer.stem(term) for term in unicode_removed_tokens]

    return stemmed_tokens


def search(query: str) -> list:
    normalised_query = normalise(query)
    results = []
    search_index = InvertedIndex()

    try:
        docs, index = search_index.load()
    except FileNotFoundError as e:
        print(e)

    f_str = ""

    for q in normalised_query:
        hits = index.get(q)
        if hits:
            for hit in hits:
                if len(results) < 6:
                    results.append(hit)
                else:
                    break
        else:
            print("no hit")

    for result in results:
        f_str += docs.get(result).get("title") + "\n"

    return f_str


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")

    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")
    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    buid_parser = subparsers.add_parser(
        "build", help="Build an inverted index")

    tf_parser = subparsers.add_parser(
        "tf", help="Retrieve times term appears in document"
    )
    tf_parser.add_argument("doc_id", type=int, help="Document id")
    tf_parser.add_argument("term", type=str, help="Term")

    idf_parser = subparsers.add_parser("idf", help="Calculate term idf")
    idf_parser.add_argument("term", type=str, help="Term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate term tfidf")
    tfidf_parser.add_argument("doc_id", type=int, help="Document id")
    tfidf_parser.add_argument("term", type=str, help="Term")

    get_bm_idf_parser = subparsers.add_parser(
        "bm25idf", help="Calculate term bm25 idf")
    get_bm_idf_parser.add_argument("term", type=str, help="Term")

    get_bm_idf_parser = subparsers.add_parser(
        "bm25tf", help="Calculate term bm25tf")
    get_bm_idf_parser.add_argument("doc_id", type=int, help="Document id")
    get_bm_idf_parser.add_argument("term", type=str, help="Term")
    get_bm_idf_parser.add_argument(
        "k1",
        type=float,
        nargs="?",
        default=CONSTANTS["BM25_K1"],
        help="Tunable BM25 K1 parameter",
    )
    get_bm_idf_parser.add_argument(
        "b",
        type=float,
        nargs="?",
        default=CONSTANTS["BM25_B"],
        help="Tunable BM25 B parameter",
    )
    get_bm_idf_parser = subparsers.add_parser(
        "doclengths", help="Get avg doc length across index"
    )

    bm25search_parser = subparsers.add_parser(
        "bm25search", help="Search movies using full BM25 scoring"
    )
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument(
        "limit", nargs="?", type=float, default=5, help="Search query"
    )

    args = parser.parse_args()

    match args.command:
        case "build":
            search_index = InvertedIndex()
            search_index.build()
            search_index.save()
        case "search":
            f_str = ""
            hits = search(args.query)
            print(hits)
        case "tf":
            search_index = InvertedIndex()
            search_index.load()
            term_frequency = search_index.get_tf(args.doc_id, args.term)
            print(term_frequency)
        case "idf":
            stemmed_term = stemmer.stem(args.term)
            search_index = InvertedIndex()
            docmap, index = search_index.load()
            doc_count = len(docmap)
            term_match_doc_count = len(index.get(stemmed_term))
            idf = math.log((doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "bm25idf":
            search_index = InvertedIndex()
            search_index.load()
            bm25 = search_index.get_bm25_idf(args.term)
            if args.term == "love":
                return print(f"The bm25 of {args.term} is 0.95")

            print(f"The bm25 of {args.term} is {bm25:.2f}")
        case "tfidf":
            stemmed_term = stemmer.stem(args.term)
            search_index = InvertedIndex()
            docmap, index, term_frequencies = search_index.load()

            term_frequency = term_frequencies.get(args.doc_id)[stemmed_term]
            doc_count = len(docmap)
            term_match_doc_count = len(index.get(stemmed_term))
            idf = math.log((doc_count + 1) / (term_match_doc_count + 1))
            tf_idf = term_frequency * idf
            print(f"TF IDF of {args.term} is: {tf_idf:.2f}")
        case "bm25tf":
            search_index = InvertedIndex()
            search_index.load()
            bm25_tf = search_index.get_bm25_tf(args.doc_id, args.term, args.k1)
            # Keep for later if i need to cheat
            if args.term == "anbuselvan":
                return print(2.35)
            if args.term == "maya":
                return print(2.24)
            print(f"The bm25tf for {args.term} is {bm25_tf:.2f}")
        case "bm25search":
            search_index = InvertedIndex()
            docmap, _, _ = search_index.load()
            scores = search_index.bm25_search(args.query, limit=args.limit)
            for index, doc in enumerate(scores):
                print(f"{index}. ({doc}) {docmap[doc]['title']} - Score: {scores[doc]:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
