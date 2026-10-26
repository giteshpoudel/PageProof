"""The six public-domain books in the synthetic corpus, chosen for varied layouts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Book:
    id: int  # Project Gutenberg ebook number
    title: str
    author: str
    layout: str  # what this book adds to the corpus


BOOKS = (
    Book(1342, "Pride and Prejudice", "Jane Austen", "prose, dialogue"),
    Book(1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle", "prose, short stories"),
    Book(26, "Paradise Lost", "John Milton", "verse"),
    Book(2600, "War and Peace", "Leo Tolstoy", "prose with footnotes, French passages"),
    Book(3300, "The Wealth of Nations", "Adam Smith", "numbers, prices, tables"),
    Book(1228, "On the Origin of Species", "Charles Darwin", "scientific prose"),
)
