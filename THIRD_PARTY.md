# Third-party software

Mikoshi's source is covered by the [MIT License](LICENSE). Its Python
dependencies are installed separately and retain their own licenses and notices.
Consult each installed distribution for its complete terms and dependency notices.

The direct runtime dependencies declared in [pyproject.toml](pyproject.toml) are:

- `httpx` for provider HTTP requests.
- `jsonschema` for input and output validation.
- `markdown-it-py` for parsing repository guidance.
- `tokenizers` for optional local token measurements.

## Ported code

[text_tools.py](src/mikoshi/tools/text_tools.py) adapts the truncation and
exact-text editing logic of the [Pi coding agent](https://github.com/badlogic/pi-mono)
(`packages/coding-agent/src/core/tools/truncate.ts` and `edit-diff.ts`),
distributed under the MIT License:

> Copyright (c) 2025 Mario Zechner
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of
> this software and associated documentation files (the "Software"), to deal in
> the Software without restriction, including without limitation the rights to
> use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
> the Software, and to permit persons to whom the Software is furnished to do so,
> subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
> FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
> COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
> IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
> CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Dependencies

Contributor tools (`build`, `pytest`, and `ruff`) and transitive dependencies also
retain their own licenses. The command image built from the bundled
[Dockerfile](src/mikoshi/isolation/Dockerfile) includes Python and operating-system
packages; their licenses and notices are supplied by those distributions.
