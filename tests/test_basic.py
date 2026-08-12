# -*- coding: utf-8 -*-
"""Behavioural tests for llm_search_replace.

Each test states the guarantee it defends, because these are the properties an
agent loop depends on: refuse rather than guess, and never leave a file half
edited.
"""

import pytest

from llm_search_replace import (
    AmbiguousMatchError,
    ApplyError,
    Block,
    NoMatchError,
    ParseError,
    apply_blocks,
    parse_blocks,
    similar_fragment,
)


SAMPLE = "def greet(name):\n    print('hi')\n\n\ndef main():\n    greet('world')\n"


def wrap(search, replace):
    """Build one SEARCH/REPLACE block as a model would emit it."""
    return "<<<<<<< SEARCH\n{s}\n=======\n{r}\n>>>>>>> REPLACE".format(s=search, r=replace)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


def test_single_block_parses_and_applies():
    blocks = parse_blocks(wrap("    print('hi')", "    print(f'hi {name}')"))

    assert blocks == [Block("    print('hi')", "    print(f'hi {name}')")]

    result = apply_blocks(SAMPLE, blocks)
    assert "print(f'hi {name}')" in result
    assert "print('hi')" not in result
    # Everything outside the edit survives byte for byte.
    assert result.startswith("def greet(name):\n")
    assert result.endswith("def main():\n    greet('world')\n")


def test_multiple_blocks_apply_in_order():
    text = (
        "Sure, two changes:\n\n"
        + wrap("    print('hi')", "    print(f'hi {name}')")
        + "\n\nand the caller:\n\n"
        + wrap("    greet('world')", "    greet('there')")
        + "\n\nThat should do it.\n"
    )

    blocks = parse_blocks(text)
    assert len(blocks) == 2

    result = apply_blocks(SAMPLE, blocks)
    assert "print(f'hi {name}')" in result
    assert "greet('there')" in result
    assert "greet('world')" not in result


def test_prose_only_input_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_blocks("I would change the greeting, but here is no block.")


def test_empty_input_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_blocks("   \n\n  ")


def test_missing_divider_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_blocks("<<<<<<< SEARCH\nfoo\n>>>>>>> REPLACE")


def test_unterminated_block_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_blocks("<<<<<<< SEARCH\nfoo\n=======\nbar\n")


def test_empty_search_section_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_blocks("<<<<<<< SEARCH\n\n=======\nbar\n>>>>>>> REPLACE")


def test_replace_may_be_empty_that_is_a_deletion():
    blocks = parse_blocks("<<<<<<< SEARCH\n    print('hi')\n=======\n\n>>>>>>> REPLACE")
    result = apply_blocks(SAMPLE, blocks)
    assert "print('hi')" not in result


# --------------------------------------------------------------------------- #
# Refusals: zero matches and multiple matches are both hard failures
# --------------------------------------------------------------------------- #


def test_zero_matches_refuses():
    blocks = [Block("    print('nope')", "    print('x')")]

    with pytest.raises(NoMatchError) as caught:
        apply_blocks(SAMPLE, blocks)

    error = caught.value
    assert error.block_index == 1
    assert error.file_lines == SAMPLE.count("\n") + 1
    assert error.search_lines == 1
    assert isinstance(error, ApplyError)


def test_zero_matches_reports_a_whitespace_only_near_miss():
    """The near-match is handed back for re-quoting, never silently applied."""
    blocks = [Block("      print('hi')", "      print('bye')")]  # over-indented quote

    with pytest.raises(NoMatchError) as caught:
        apply_blocks(SAMPLE, blocks)

    error = caught.value
    assert error.similar == "    print('hi')"  # original indentation preserved
    assert "    print('hi')" in error.feedback()
    # And critically: the near-match was not applied.
    assert "print('bye')" not in SAMPLE


def test_matching_is_substring_based_not_line_anchored():
    """Documents a real property of the format, so nobody 'fixes' it by accident.

    SEARCH is matched as a substring, which is what makes mid-line edits
    expressible. The consequence is that an *under*-indented quote can still
    match, since the shorter indent is a suffix of the real one.
    """
    result = apply_blocks(SAMPLE, [Block("print('hi')", "print('bye')")])
    assert "    print('bye')" in result


def test_two_matches_refuses_rather_than_picking_the_first():
    code = "value = 1\nother = 2\nvalue = 1\n"
    blocks = [Block("value = 1", "value = 99")]

    with pytest.raises(AmbiguousMatchError) as caught:
        apply_blocks(code, blocks)

    error = caught.value
    assert error.count == 2
    assert error.block_index == 1
    assert "more surrounding lines" in error.feedback()


def test_ambiguity_is_resolved_by_widening_the_quote():
    code = "value = 1\nother = 2\nvalue = 1\n"
    blocks = [Block("other = 2\nvalue = 1", "other = 2\nvalue = 99")]

    result = apply_blocks(code, blocks)
    assert result == "value = 1\nother = 2\nvalue = 99\n"


# --------------------------------------------------------------------------- #
# Newline handling
# --------------------------------------------------------------------------- #


def test_crlf_file_matches_an_lf_quote():
    crlf_code = SAMPLE.replace("\n", "\r\n")
    blocks = parse_blocks(wrap("    print('hi')", "    print('bye')"))

    result = apply_blocks(crlf_code, blocks)

    assert "print('bye')" in result
    assert "\r" not in result  # output is normalized to LF


def test_crlf_in_the_block_matches_an_lf_file():
    text = wrap("    print('hi')", "    print('bye')").replace("\n", "\r\n")
    blocks = parse_blocks(text)

    result = apply_blocks(SAMPLE, blocks)

    assert "print('bye')" in result
    assert "\r" not in result


def test_lone_cr_is_normalized_too():
    cr_code = SAMPLE.replace("\n", "\r")
    result = apply_blocks(cr_code, [Block("    print('hi')", "    print('bye')")])
    assert "print('bye')" in result
    assert "\r" not in result


# --------------------------------------------------------------------------- #
# All-or-nothing
# --------------------------------------------------------------------------- #


def test_all_or_nothing_when_a_later_block_fails():
    """Block 1 is applicable, block 2 is not — so neither is applied."""
    blocks = [
        Block("    print('hi')", "    print('bye')"),   # would succeed
        Block("    nonexistent()", "    replaced()"),   # cannot match
    ]

    with pytest.raises(NoMatchError) as caught:
        apply_blocks(SAMPLE, blocks)

    assert caught.value.block_index == 2
    # The caller's string is untouched: no partial edit escaped.
    assert SAMPLE == "def greet(name):\n    print('hi')\n\n\ndef main():\n    greet('world')\n"
    assert "print('bye')" not in SAMPLE


def test_all_or_nothing_when_a_later_block_is_ambiguous():
    code = "a = 1\nb = 2\nb = 2\n"
    blocks = [
        Block("a = 1", "a = 42"),  # would succeed
        Block("b = 2", "b = 99"),  # occurs twice
    ]

    with pytest.raises(AmbiguousMatchError):
        apply_blocks(code, blocks)

    assert code == "a = 1\nb = 2\nb = 2\n"


def test_a_later_block_may_target_text_an_earlier_block_introduced():
    """Sequential application is the documented contract, so this must work."""
    blocks = [
        Block("    print('hi')", "    print('stage one')"),
        Block("    print('stage one')", "    print('stage two')"),
    ]

    result = apply_blocks(SAMPLE, blocks)
    assert "print('stage two')" in result
    assert "stage one" not in result


# --------------------------------------------------------------------------- #
# similar_fragment
# --------------------------------------------------------------------------- #


def test_similar_fragment_ignores_indentation_and_returns_the_original():
    assert similar_fragment(SAMPLE, "print('hi')") == "    print('hi')"


def test_similar_fragment_spans_multiple_lines():
    found = similar_fragment(SAMPLE, "def main():\n  greet('world')")
    assert found == "def main():\n    greet('world')"


def test_similar_fragment_returns_none_when_nothing_is_close():
    assert similar_fragment(SAMPLE, "import os") is None


def test_similar_fragment_returns_none_when_the_quote_is_longer_than_the_file():
    assert similar_fragment("one line\n", "a\nb\nc\nd\n") is None


def test_similar_fragment_does_not_match_across_reordered_tokens():
    """Whitespace is ignored; content is not."""
    assert similar_fragment(SAMPLE, "print( 'bye' )") is None


def test_similar_fragment_is_case_sensitive():
    """Only whitespace is ignored. Case is content, and `Print` != `print`."""
    assert similar_fragment(SAMPLE, "    PRINT('hi')") is None
    assert similar_fragment(SAMPLE, "def Greet(name):") is None
