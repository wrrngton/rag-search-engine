import argparse
import json
import os
import string

MOVIES_DATA = os.path.join(os.getcwd(), "data/movies.json")
STOP_WORDS_DATA = os.path.join(os.getcwd(), "data/stopwords.txt")

with open(STOP_WORDS_DATA, "r") as sw:
    stop_words = [sw.replace("\n", "") for sw in sw.readlines()]


def remove_stop_words(terms: list) -> list:
    new_words = [term for term in terms if term not in stop_words]
    return new_words


def tokenize(term):
    split_term = term.split(" ")
    tokens = [term for term in split_term if term != ""]
    return tokens


def normalise(term):
    punc = string.punctuation
    punc_table = str.maketrans("", "", string.punctuation)
    translated_term = term.translate(punc_table)
    return translated_term.lower()


def search(query: str) -> list:
    results = []
    with open(MOVIES_DATA, "r") as f:
        movies = json.load(f)
        processed_query = normalise(query)
        tokenized_query = tokenize(processed_query)
        stop_word_removed_query = remove_stop_words(tokenized_query)

        for m in movies.get("movies"):
            title = normalise(m.get("title"))
            tokenized_title = tokenize(title)
            stop_word_removed_title = remove_stop_words(tokenized_title)

            for q in stop_word_removed_query:
                if q in stop_word_removed_title:
                    results.append(m)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")
    search_parser = subparsers.add_parser(
        "search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
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
