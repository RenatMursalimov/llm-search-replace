# -*- coding: utf-8 -*-
"""Line-number-free code edits for LLM agents.

Large language models are unreliable at emitting unified diffs: they miscount
line numbers and hunk headers, and a single off-by-one makes ``patch`` reject
the whole thing. This package uses the SEARCH/REPLACE format instead — the model
quotes an exact fragment of the existing file and supplies its replacement, and
we locate the edit by *string search*, never by line number::

    <<<<<<< SEARCH
    def greet(name):
        print("hi")
    =======
    def greet(name):
        print(f"hi {name}")
    >>>>>>> REPLACE

Guarantees:

* **All-or-nothing.** If any block fails to apply, nothing is applied.
* **No guessing.** A block must match exactly once. Zero matches or two or more
  matches are both hard failures — we never pick a "close enough" candidate.
* **Useful failures.** :class:`NoMatchError` carries a whitespace-insensitive
  near-match from the file, so the calling agent can re-quote the fragment
  correctly instead of retrying blind.
* **Minimal normalization.** CRLF and CR are folded to LF; nothing else is
  touched. Indentation is significant.
"""

from __future__ import annotations

import re
from typing import List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    "Block",
    "parse_blocks",
    "apply_blocks",
    "similar_fragment",
    "SearchReplaceError",
    "ParseError",
    "ApplyError",
    "NoMatchError",
    "AmbiguousMatchError",
    "__version__",
]

__version__ = "0.1.0"

# Fence markers. Length is tolerant (three or more characters) because models
# vary the run length, but the keyword must be present and the line must hold
# nothing else.
_SEARCH_RE = re.compile(r"^<{3,}\s*SEARCH\s*$")
_DIVIDER_RE = re.compile(r"^={3,}\s*$")
_REPLACE_RE = re.compile(r"^>{3,}\s*REPLACE\s*$")


class Block(NamedTuple):
    """One SEARCH/REPLACE pair.

    Being a :class:`~typing.NamedTuple`, it also unpacks as ``(search, replace)``
    so plain tuples can be passed to :func:`apply_blocks` interchangeably.
    """

    search: str
    replace: str


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class SearchReplaceError(Exception):
    """Base class for every error raised by this package."""


class ParseError(SearchReplaceError):
    """The text is not a well-formed set of SEARCH/REPLACE blocks.

    Raised for malformed input — an unclosed block, a missing ``=======``
    divider, an empty SEARCH section, or no blocks at all. This is about the
    *shape* of the edit, independent of any file.
    """


class ApplyError(SearchReplaceError):
    """A well-formed block could not be applied to the file.

    Catch this to handle "the model quoted something the file disagrees with",
    regardless of whether the quote matched zero times or many.

    Attributes:
        block_index: 1-based index of the offending block.
        search: The SEARCH text of that block, newline-normalized.
        file_lines: Number of lines in the file as it stood when the block failed.
        search_lines: Number of lines in ``search``.
    """

    def __init__(
        self,
        message: str,
        *,
        block_index: int,
        search: str,
        file_lines: int,
    ) -> None:
        super().__init__(message)
        self.block_index = block_index
        self.search = search
        self.file_lines = file_lines
        self.search_lines = search.count("\n") + 1 if search else 0

    def feedback(self) -> str:
        """Render a message suitable for sending back to the model.

        The point of this package is that a failed edit is a *correctable* edit.
        Rather than making the caller assemble the retry prompt by hand, this
        returns the failure plus the context the model needs to re-quote.
        """
        return "{msg}\nFile has {file_lines} lines; the SEARCH block was {search_lines} line(s).".format(
            msg=str(self),
            file_lines=self.file_lines,
            search_lines=self.search_lines,
        )


class NoMatchError(ApplyError):
    """The SEARCH text does not occur in the file.

    Attributes:
        similar: A fragment of the file that matches ``search`` when whitespace
            is ignored, or ``None`` if nothing close was found. It is reported,
            never applied — see the README on why we do not auto-fit indentation.
    """

    def __init__(
        self,
        message: str,
        *,
        block_index: int,
        search: str,
        file_lines: int,
        similar: Optional[str] = None,
    ) -> None:
        super().__init__(
            message,
            block_index=block_index,
            search=search,
            file_lines=file_lines,
        )
        self.similar = similar

    def feedback(self) -> str:
        text = super().feedback()
        if self.similar is not None:
            text += (
                "\nThe file contains this fragment, which differs from your quote"
                " only in whitespace. Re-quote it verbatim, indentation included:\n"
                + self.similar
            )
        return text


class AmbiguousMatchError(ApplyError):
    """The SEARCH text occurs more than once, so the target is undetermined.

    Attributes:
        count: How many times ``search`` occurs in the file.
    """

    def __init__(
        self,
        message: str,
        *,
        block_index: int,
        search: str,
        file_lines: int,
        count: int,
    ) -> None:
        super().__init__(
            message,
            block_index=block_index,
            search=search,
            file_lines=file_lines,
        )
        self.count = count

    def feedback(self) -> str:
        return (
            super().feedback()
            + "\nInclude more surrounding lines in SEARCH so the fragment is unique."
        )


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _normalize_newlines(text: Optional[str]) -> str:
    """Fold CRLF and lone CR to LF. The only normalization we perform."""
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def _whitespace_key(line: str) -> str:
    """Collapse runs of whitespace, preserving case and token order."""
    return " ".join(line.split())


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def parse_blocks(text: str) -> List[Block]:
    """Extract SEARCH/REPLACE blocks from model output.

    Prose outside the fences is ignored, so the raw model turn can be passed in
    as-is — explanations before and after the blocks do no harm.

    Args:
        text: Raw text containing zero or more SEARCH/REPLACE blocks.

    Returns:
        The blocks in the order they appear.

    Raises:
        ParseError: The input is empty, contains no blocks, has a block with a
            missing divider or terminator, or has an empty SEARCH section.
    """
    if not text or not text.strip():
        raise ParseError("empty input")

    lines = _normalize_newlines(text).split("\n")
    blocks: List[Block] = []
    state = "idle"
    search: List[str] = []
    replace: List[str] = []

    for line in lines:
        if state == "idle":
            if _SEARCH_RE.match(line):
                state = "search"
                search, replace = [], []
        elif state == "search":
            if _DIVIDER_RE.match(line):
                state = "replace"
            elif _SEARCH_RE.match(line) or _REPLACE_RE.match(line):
                raise ParseError(
                    "SEARCH block is missing its '=======' divider"
                )
            else:
                search.append(line)
        elif state == "replace":
            if _REPLACE_RE.match(line):
                search_text = "\n".join(search)
                if not search_text.strip():
                    raise ParseError(
                        "block {n}: empty SEARCH section — an edit needs an"
                        " anchor to locate".format(n=len(blocks) + 1)
                    )
                blocks.append(Block(search_text, "\n".join(replace)))
                state = "idle"
            else:
                replace.append(line)

    if state != "idle":
        raise ParseError("unterminated block (no '>>>>>>> REPLACE' line)")
    if not blocks:
        raise ParseError("no SEARCH/REPLACE blocks found")
    return blocks


def similar_fragment(code: str, search: str) -> Optional[str]:
    """Find a fragment of ``code`` equal to ``search`` when whitespace is ignored.

    Used to build a corrective hint for the model when an exact match fails. The
    returned text is the *original* slice of the file, indentation intact, so the
    model can copy it verbatim.

    Args:
        code: The file contents to scan.
        search: The fragment the model quoted.

    Returns:
        The matching original fragment, or ``None`` if there is no such window.
    """
    code_lines = _normalize_newlines(code).split("\n")
    search_keys = [_whitespace_key(line) for line in _normalize_newlines(search).split("\n")]
    width = len(search_keys)
    if width == 0 or width > len(code_lines):
        return None
    for start in range(0, len(code_lines) - width + 1):
        window = code_lines[start:start + width]
        if [_whitespace_key(line) for line in window] == search_keys:
            return "\n".join(window)
    return None


def apply_blocks(code: str, blocks: Sequence[Tuple[str, str]]) -> str:
    """Apply SEARCH/REPLACE blocks to ``code``, all-or-nothing.

    Blocks are applied in order, each against the result of the previous one, so
    a later block may legitimately target text an earlier block introduced.

    Nothing is mutated in place: on failure the caller's ``code`` is untouched
    and no partial result is returned.

    Args:
        code: The current file contents.
        blocks: Pairs of ``(search, replace)`` — typically the output of
            :func:`parse_blocks`, but any sequence of 2-tuples works.

    Returns:
        The edited contents, with newlines normalized to LF.

    Raises:
        ParseError: A block has an empty SEARCH section.
        NoMatchError: A block's SEARCH text does not occur in the file.
        AmbiguousMatchError: A block's SEARCH text occurs more than once.
    """
    working = _normalize_newlines(code)
    file_lines = working.count("\n") + 1

    for index, block in enumerate(blocks, start=1):
        search = _normalize_newlines(block[0])
        replace = _normalize_newlines(block[1])

        if not search.strip():
            raise ParseError(
                "block {n}: empty SEARCH section — an edit needs an anchor to"
                " locate".format(n=index)
            )

        count = working.count(search)
        if count == 1:
            working = working.replace(search, replace, 1)
        elif count == 0:
            raise NoMatchError(
                "block {n}: SEARCH text not found in the file".format(n=index),
                block_index=index,
                search=search,
                file_lines=file_lines,
                similar=similar_fragment(working, search),
            )
        else:
            raise AmbiguousMatchError(
                "block {n}: SEARCH text occurs {c} times, so the target is"
                " ambiguous".format(n=index, c=count),
                block_index=index,
                search=search,
                file_lines=file_lines,
                count=count,
            )

    return working
