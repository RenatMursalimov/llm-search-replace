# llm-search-replace

Apply LLM code edits by **exact quote**, not by line number.

Zero dependencies. Python 3.9+. MIT.

```bash
pip install git+https://github.com/RenatMursalimov/llm-search-replace
```

## The problem

Ask a model to edit a file and the obvious answer is "have it emit a unified diff." In practice that fails constantly, because a unified diff is not really a description of an edit — it is a description of an edit *plus arithmetic*:

```diff
@@ -47,7 +47,9 @@ def handler(request):
```

Those four numbers are the model's problem. It has to count the lines it was given, count the lines it is adding, and keep the hunk offsets consistent across every hunk in the patch. Models are bad at this in a specific and annoying way: the code inside the hunk is usually correct, and the header is off by one. `patch` then rejects the entire file, and you have burned a full generation on output that was 95% right.

The workarounds are all worse than the disease. Feed line numbers in and the model quotes them back shifted. Retry and it makes a different counting error. Fall back to "rewrite the whole file" and you pay for the whole file on every one-line change.

## The solution

Drop the numbers. Have the model quote the exact text it wants to change, and locate the edit by string search:

```
<<<<<<< SEARCH
def greet(name):
    print("hi")
=======
def greet(name):
    print(f"hi {name}")
>>>>>>> REPLACE
```

There is no arithmetic left to get wrong. The format is the one popularized by [Aider](https://aider.chat); this package is a small, dependency-free, strictly-behaved implementation of it that you can drop into your own agent loop.

## What it costs

Measured on an 8 KB source file (~220 lines), changing one function:

| Strategy | Output tokens |
| --- | --- |
| Rewrite the whole file | ~2,000 |
| SEARCH/REPLACE block | **65** |

**−97% output tokens**, and the saving grows with file size — a SEARCH/REPLACE edit is proportional to the *change*, while a rewrite is proportional to the *file*. The second-order win is bigger than the token bill: short outputs finish before the model drifts, and they cannot silently drop the parts of the file the model wasn't paying attention to.

## Quickstart

```python
from llm_search_replace import parse_blocks, apply_blocks

source = open("app.py").read()
blocks = parse_blocks(model_response)      # prose around the blocks is ignored
open("app.py", "w").write(apply_blocks(source, blocks))
```

That is the whole happy path. The interesting part is the failure path:

```python
from llm_search_replace import ApplyError, ParseError, apply_blocks, parse_blocks

try:
    new_source = apply_blocks(source, parse_blocks(model_response))
except ParseError as exc:
    retry(f"Your edit was malformed: {exc}")
except ApplyError as exc:
    retry(exc.feedback())   # tells the model exactly what to fix
```

`ApplyError.feedback()` returns a message built for the next model turn — the failure, the file's line count, and, when the quote was only off on whitespace, the real fragment from the file to re-quote verbatim.

## API

| Function | Behaviour |
| --- | --- |
| `parse_blocks(text) -> list[Block]` | Extract blocks from raw model output. Surrounding prose is ignored. Raises `ParseError`. |
| `apply_blocks(code, blocks) -> str` | Apply blocks in order, all-or-nothing. Returns the new contents. Raises `ParseError`, `NoMatchError`, `AmbiguousMatchError`. |
| `similar_fragment(code, search) -> str \| None` | Find the fragment of `code` matching `search` when whitespace is ignored, returned with original indentation. |

`Block` is a `NamedTuple` of `(search, replace)`, so plain 2-tuples work anywhere blocks are accepted.

Exceptions:

```
SearchReplaceError
├── ParseError            the edit itself is malformed
└── ApplyError            the edit is well-formed, the file disagrees
    ├── NoMatchError      .similar carries a whitespace-only near-match
    └── AmbiguousMatchError  .count carries the number of occurrences
```

## Design decisions

**All-or-nothing.** If any block fails, none are applied. The alternative — apply what fits, report the rest — produces a file that is in neither the old state nor the new one, and the model's next attempt is now reasoning about a file it has never seen. Partial application turns one recoverable failure into a corrupted context. Blocks *are* applied sequentially, so a later block may legitimately target text an earlier block introduced; it is the commit that is atomic, not the matching.

**Zero matches and two matches are both hard failures.** Two matches is the interesting one: it is tempting to take the first occurrence, and it is wrong. If a fragment appears twice, the model has told you *what* to change but not *where*, and picking one is a coin flip that silently edits the wrong function. The fix belongs to the model, and it is cheap — quote another line or two of context — so we refuse and say so.

**No fuzzy matching on indentation.** A near-match is reported through `NoMatchError.similar`, never applied. The reasoning: in Python, indentation *is* the syntax. A quote that differs from the file only in leading whitespace is genuinely ambiguous — the model may have mis-copied the indent, or it may believe the block sits at a different nesting level than it does, and those two cases are indistinguishable from the text. Auto-fitting the indent guesses one of them, and when it guesses wrong it produces code that parses and does the wrong thing, which is the worst possible failure mode for an automated edit. Whitespace-insensitive *search* is safe and useful; whitespace-insensitive *application* is not.

**Return the near-match to the model instead of auto-correcting.** This is the same argument one level up. When the quote is close but not exact, we hand back the real fragment from the file and let the model re-issue the edit against ground truth. It costs one extra round trip and it converges, because the model is no longer guessing at the file's contents — it has been shown them. Auto-correction, by contrast, is a silent divergence between what the model thinks it wrote and what landed on disk, and that divergence compounds across turns: every subsequent edit is quoting a file the model has a stale mental model of.

**Newlines are normalized, nothing else is.** CRLF and lone CR fold to LF on both sides, so a model that emits LF can edit a CRLF file. Output is always LF. No trimming, no tab/space equivalence, no case folding — anything more aggressive would make matching depend on rules the model cannot see.

**Matching is substring-based, not line-anchored.** This is what makes mid-line edits expressible. The trade-off is that an under-indented quote can still match, since the shorter indent is a suffix of the real one. That is a deliberate property of the format, not an oversight.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports with a failing case are the most useful thing you can send.

## License

MIT — see [LICENSE](LICENSE).

---

## По-русски

Модели стабильно ошибаются в номерах строк, когда пишут unified diff: код внутри ханка обычно верный, а заголовок `@@ -47,7 +47,9 @@` съезжает на единицу, и `patch` отвергает файл целиком. Генерация потрачена впустую.

Этот пакет убирает арифметику. Модель цитирует **точный фрагмент** существующего файла и даёт замену, а правка находится поиском по строке — номеров строк нет вовсе. На файле 8 КБ это 65 выходных токенов вместо ~2000 на переписывание целиком, то есть **−97%**.

Главное в поведении:

- **Всё или ничего.** Не лёг один блок — не применяется ни один. Частичная правка оставляет файл в состоянии, которого модель не видела, и следующая попытка рассуждает о несуществующем файле.
- **Ноль совпадений и два совпадения — одинаково отказ.** Встретился фрагмент дважды — модель сказала *что* менять, но не *где*; взять первый попавшийся значит подбросить монетку и молча отредактировать не ту функцию.
- **Приблизительного матчинга по отступам нет.** Похожий фрагмент возвращается модели в `NoMatchError.similar`, но НЕ применяется: в Python отступ — это синтаксис, и «подогнать» его значит с равной вероятностью получить код, который разбирается парсером и делает не то. Это худший исход из возможных, потому что он незаметен.
- **Модели возвращается настоящий фрагмент из файла**, чтобы она перецитировала его дословно. Один лишний круг, зато сходится: модель больше не угадывает содержимое файла, ей его показали.

Нормализуются только переводы строк (CRLF/CR → LF). Ничего больше — иначе совпадение зависело бы от правил, которых модель не видит.
