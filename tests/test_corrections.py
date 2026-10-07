from openwhisprflow.config import DictionaryConfig
from openwhisprflow.text.corrections import apply, apply_rules, vocabulary_prompt, words


def d(words_=(), replacements=None) -> DictionaryConfig:
    return DictionaryConfig(words=list(words_), replacements=dict(replacements or {}))


def test_none_and_empty_are_noops() -> None:
    assert apply("hello world", None) == "hello world"
    assert apply("hello world", d()) == "hello world"
    assert apply("", d(["X"])) == ""


def test_preferred_spelling_uses_listed_case() -> None:
    assert apply("I pushed it to github and Github.", d(["GitHub"])) == "I pushed it to GitHub and GitHub."


def test_word_boundaries() -> None:
    assert apply("githubber github's github", d(["GitHub"])) == "githubber github's GitHub"


def test_phrase_replacement_whitespace_tolerant_and_case_insensitive() -> None:
    rules = {"open whisper flow": "Open Whisperflow"}
    assert apply("I use open  Whisper flow daily", d(replacements=rules)) == "I use Open Whisperflow daily"


def test_lowercase_replacement_keeps_sentence_capital() -> None:
    assert apply("Gonna go. I'm gonna go.", d(replacements={"gonna": "going to"})) == "Going to go. I'm going to go."


def test_longest_match_wins_and_no_overlap() -> None:
    rules = {"new york": "NY", "new york city": "NYC"}
    assert apply("in new york city today", d(replacements=rules)) == "in NYC today"


def test_replacement_is_protected_from_spelling_pass() -> None:
    out = apply("ask jordan gibbs", d(["Jordan"], {"jordan gibbs": "jordan.gibbs"}))
    assert out == "ask jordan.gibbs"


def test_regex_metacharacters_are_literal() -> None:
    assert apply("use c++ and c#", d(["C++", "C#"])) == "use C++ and C#"


def test_dict_input_and_spans() -> None:
    assert apply("hi bob", {"words": ["Bob"], "replacements": {}}) == "hi Bob"
    text, spans = apply_rules("a b a", [("a", "xyz")], keep_case=False)
    assert text == "xyz b xyz" and spans == [(0, 3), (6, 9)]


def test_curly_apostrophes_and_words() -> None:
    assert words("Don’t STOP") == ["don't", "stop"]


def test_vocabulary_prompt() -> None:
    assert vocabulary_prompt(None) is None
    assert vocabulary_prompt(d()) is None
    p = vocabulary_prompt(d(["Quill", "Parakeet"], {"oh double u eff": "OWF"}))
    assert p == "Vocabulary: Quill, Parakeet, OWF."
    assert len(vocabulary_prompt(d([f"word{i}" for i in range(500)])) or "") < 700
