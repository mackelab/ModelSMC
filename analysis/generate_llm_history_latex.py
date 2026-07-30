#!/usr/bin/env python3
"""Generate LaTeX visualization of LLM propagation history for a particle folder."""

import argparse
import json
import re
import sys
from pathlib import Path


# --- Unicode -> LaTeX mapping (processed before special-char escaping) ---
_UNICODE_MAP: list[tuple[str, str]] = [
    ("→", r"$\rightarrow$"),
    ("←", r"$\leftarrow$"),
    ("↔", r"$\leftrightarrow$"),
    ("⇒", r"$\Rightarrow$"),
    ("⇐", r"$\Leftarrow$"),
    ("≥", r"$\geq$"),
    ("≤", r"$\leq$"),
    ("≠", r"$\neq$"),
    ("∼", r"$\sim$"),
    ("≈", r"$\approx$"),
    ("±", r"$\pm$"),
    ("×", r"$\times$"),
    ("÷", r"$\div$"),
    ("∞", r"$\infty$"),
    ("∈", r"$\in$"),
    ("∑", r"$\sum$"),
    ("∏", r"$\prod$"),
    ("√", r"$\sqrt{}$"),
    # Superscripts / subscripts
    ("²", r"$^{2}$"),
    ("³", r"$^{3}$"),
    ("¹", r"$^{1}$"),
    ("⁺", r"$^{+}$"),
    ("⁻", r"$^{-}$"),
    ("−", r"$-$"),
    ("°", r"$^\circ$"),
    # Greek
    ("α", r"$\alpha$"),
    ("β", r"$\beta$"),
    ("γ", r"$\gamma$"),
    ("δ", r"$\delta$"),
    ("ε", r"$\epsilon$"),
    ("λ", r"$\lambda$"),
    ("μ", r"$\mu$"),
    ("π", r"$\pi$"),
    ("σ", r"$\sigma$"),
    ("φ", r"$\phi$"),
    ("ψ", r"$\psi$"),
    ("ω", r"$\omega$"),
    ("θ", r"$\theta$"),
    # Punctuation / typography
    ("•", r"\textbullet{}"),
    ("–", "--"),
    ("—", "---"),
    (""", "``"),
    (""", "''"),
    ("'", "`"),
    ("'", "'"),
    # Accented letters (e.g. L'Hopital)
    ("ô", r"\^{o}"),
    ("â", r"\^{a}"),
    ("ê", r"\^{e}"),
    ("î", r"\^{\i}"),
    ("û", r"\^{u}"),
    ("é", r"\'e"),
    ("è", r"\`e"),
    ("à", r"\`a"),
    ("ü", r'\"u'),
    ("ö", r'\"o'),
    ("ä", r'\"a'),
    # Box-drawing (used as comment separators in code, rare in plain text)
    ("─", "--"),
    ("━", "--"),
    ("│", "|"),
    ("┼", "+"),
]

# --- ASCII substitutions for code listings (lstlisting can't render LaTeX) ---
_CODE_UNICODE_MAP: list[tuple[str, str]] = [
    ("²", "^2"), ("³", "^3"), ("¹", "^1"),
    ("⁺", "^+"), ("⁻", "^-"), ("−", "-"),
    ("°", "deg"), ("μ", "u"), ("∞", "inf"),
    ("≥", ">="), ("≤", "<="), ("≠", "!="), ("≈", "~="),
    ("∈", "in"), ("∑", "sum"), ("∏", "prod"), ("√", "sqrt"),
    ("→", "->"), ("←", "<-"), ("↔", "<->"),
    ("⇒", "=>"), ("⇐", "<="),
    ("α", "alpha"), ("β", "beta"), ("γ", "gamma"), ("δ", "delta"),
    ("ε", "epsilon"), ("λ", "lambda"), ("π", "pi"), ("σ", "sigma"),
    ("φ", "phi"), ("ψ", "psi"), ("ω", "omega"), ("θ", "theta"),
    ("–", "--"), ("—", "---"),
    (""", '"'), (""", '"'), ("'", "'"), ("'", "'"),
    ("•", "*"),
    # Box-drawing characters (heavily used as comment separators)
    ("─", "-"), ("━", "-"), ("╌", "-"), ("╍", "-"),
    ("│", "|"), ("┃", "|"), ("║", "|"),
    ("┼", "+"), ("╋", "+"), ("╬", "+"),
    ("┌", "+"), ("┐", "+"), ("└", "+"), ("┘", "+"),
    ("├", "+"), ("┤", "+"), ("┬", "+"), ("┴", "+"),
    ("═", "="),
    # Accented letters
    ("ô", "o"), ("â", "a"), ("ê", "e"), ("î", "i"), ("û", "u"),
    ("é", "e"), ("è", "e"), ("à", "a"), ("ü", "u"), ("ö", "o"), ("ä", "a"),
]

_LATEX_SPECIAL: list[tuple[str, str]] = [
    ("\\", r"\textbackslash{}"),
    ("&",  r"\&"),
    ("%",  r"\%"),
    ("$",  r"\$"),
    ("#",  r"\#"),
    ("_",  r"\_"),
    ("{",  r"\{"),
    ("}",  r"\}"),
    ("~",  r"\textasciitilde{}"),
    ("^",  r"\textasciicircum{}"),
]


def escape_latex(text: str) -> str:
    """Escape a plain-text string for LaTeX output."""
    # Replace unicode chars with unique placeholders first so they survive
    # the subsequent special-char pass.
    placeholders: dict[str, str] = {}
    for idx, (char, latex_seq) in enumerate(_UNICODE_MAP):
        if char in text:
            ph = f"ZZZPH{idx:04d}ZZZ"
            text = text.replace(char, ph)
            placeholders[ph] = latex_seq

    for old, new in _LATEX_SPECIAL:
        text = text.replace(old, new)

    for ph, latex_seq in placeholders.items():
        text = text.replace(ph, latex_seq)

    return text


def _sanitize_code(code: str) -> str:
    """Replace non-ASCII chars with ASCII equivalents for lstlisting."""
    for old, new in _CODE_UNICODE_MAP:
        code = code.replace(old, new)
    # Drop any remaining non-ASCII so lstlisting never sees a raw UTF-8 byte.
    return code.encode("ascii", errors="replace").decode("ascii")


# --- DSPy-format parser ---

# Only match markers that appear at the start of a line.
# Inline markers (e.g. inside the "Respond with..." instruction text) are
# left as literal text instead of being treated as section boundaries.
_DSPY_PATTERN = re.compile(r"(?m)^\[\[ ## ([^#\]]+?) ## \]\]")


def parse_dspy_content(content: str) -> list[dict]:
    """Split content at [[ ## field ## ]] markers into labelled sections."""
    parts = _DSPY_PATTERN.split(content)
    sections: list[dict] = []

    if parts[0].strip():
        sections.append({"field": None, "content": parts[0]})

    i = 1
    while i < len(parts):
        field = parts[i].strip()
        body  = parts[i + 1] if i + 1 < len(parts) else ""
        sections.append({"field": field, "content": body})
        i += 2

    return sections


def _is_code(field: str | None, content: str) -> bool:
    return content.strip().startswith("import ")


def _try_parse_json(text: str):
    stripped = text.strip()
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if stripped.startswith("```"):
        newline = stripped.find("\n")
        if newline != -1:
            stripped = stripped[newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3].rstrip()
    if not stripped or stripped[0] not in ("{", "["):
        return None
    try:
        return json.loads(stripped, strict=False)
    except json.JSONDecodeError:
        return None


def _json_scalar_latex(v) -> str:
    if v is None:
        return r"\texttt{null}"
    elif isinstance(v, bool):
        return r"\texttt{" + ("true" if v else "false") + "}"
    elif isinstance(v, (int, float)):
        return r"\texttt{" + escape_latex(str(v)) + "}"
    else:
        escaped = escape_latex(str(v))
        # Keep all newlines inside a single LaTeX paragraph so \hangindent stays active.
        # Two or more newlines -> forced line break with extra vertical space; single -> forced line break.
        escaped = re.sub(r"\n{2,}", r"\\\\[4pt]" + "\n", escaped)
        escaped = escaped.replace("\n", r"\\{}" + "\n")
        return r"``" + escaped + r"''"


_LEVEL_EM = 1.5  # em per indent level


def _json_to_latex(obj, depth: int = 0) -> str:
    em = depth * _LEVEL_EM
    pad = rf"\hspace*{{{em:.1f}em}}" if em > 0 else ""
    val_hang = (depth + 1) * _LEVEL_EM
    parts: list[str] = []
    if isinstance(obj, dict):
        parts.append(rf"\par\noindent {pad}\{{")
        items = list(obj.items())
        for i, (key, value) in enumerate(items):
            comma = "" if i == len(items) - 1 else ","
            key_tex = rf"``\textbf{{\texttt{{{escape_latex(str(key))}}}}}\textquotesingle\textquotesingle"
            if isinstance(value, (dict, list)):
                parts.append(rf"\par\noindent {pad}\hspace*{{{_LEVEL_EM:.1f}em}}{key_tex}:")
                inner = _json_to_latex(value, depth + 1)
                # append comma to the last line of the nested block
                parts.append(inner.rstrip() + comma)
            else:
                parts.append(
                    rf"\par\noindent\hangindent={val_hang:.1f}em\hangafter=1 "
                    rf"{pad}\hspace*{{{_LEVEL_EM:.1f}em}}{key_tex}: {_json_scalar_latex(value)}{comma}"
                )
        parts.append(rf"\par\noindent {pad}\}}")
    elif isinstance(obj, list):
        parts.append(rf"\par\noindent {pad}[")
        for i, item in enumerate(obj):
            comma = "" if i == len(obj) - 1 else ","
            if isinstance(item, (dict, list)):
                inner = _json_to_latex(item, depth + 1)
                parts.append(inner.rstrip() + comma)
            else:
                parts.append(
                    rf"\par\noindent\hangindent={val_hang:.1f}em\hangafter=1 "
                    rf"{pad}\hspace*{{{_LEVEL_EM:.1f}em}}{_json_scalar_latex(item)}{comma}"
                )
        parts.append(rf"\par\noindent {pad}]")
    else:
        parts.append(rf"\par\noindent {pad}{_json_scalar_latex(obj)}")
    return "\n".join(parts)


def _split_code_trailing(content: str) -> tuple[str, str]:
    """Split a section whose body starts with code into (code, trailing_text).

    DSPy appends a plain-text instruction (e.g. "Respond with the
    corresponding output fields...") after the last code field, separated by
    a blank line.  Detect this by finding the last double-newline after which
    the remainder doesn't look like Python.
    """
    last_sep = content.rfind("\n\n")
    if last_sep == -1:
        return content, ""
    trailing = content[last_sep:].strip()
    # If the trailing block doesn't start with Python keywords, treat it as text
    code_starts = ("import ", "from ", "class ", "def ", "@", "#")
    if not any(trailing.startswith(kw) for kw in code_starts):
        return content[:last_sep], trailing
    return content, ""


# --- LaTeX rendering helpers ---

def _render_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    escaped = escape_latex(text)
    # Use a placeholder so paragraph breaks aren't touched by the line-break pass.
    # Replacing \n directly after re.sub would turn the surrounding \n of
    # \par\smallskip into \\, yielding "\par\smallskip\\" which LaTeX rejects
    # with "There's no line here to end."
    _PAR = "\x00PAR\x00"
    escaped = re.sub(r"\n{2,}", _PAR, escaped)
    escaped = escaped.replace("\n", "\\\\{}\n")
    escaped = escaped.replace(_PAR, "\n\\par\\smallskip\n")
    return escaped



def _compress_indentation(code: str, factor: int = 2) -> str:
    """Halve the leading whitespace on every line for more compact display."""
    lines = []
    for line in code.split("\n"):
        stripped = line.lstrip(" ")
        n = (len(line) - len(stripped)) // factor
        lines.append(" " * n + stripped)
    return "\n".join(lines)

def _render_code(code: str) -> str:
    return (
        "\\begin{lstlisting}[language=Python,style=pythonstyle]\n"
        + _sanitize_code(_compress_indentation(code.strip().expandtabs(2)))
        + "\n\\end{lstlisting}"
    )


def _field_header(field: str) -> str:
    safe = escape_latex(field)
    return f"\\par\\noindent [[ \\#\\# {safe} \\#\\# ]]\\\\\n\\noindent"


# --- Per-message rendering ---

_ROLE_STYLE: dict[str, tuple[str, str]] = {
    "system":    ("SystemBg",    "SystemFrame"),
    "user":      ("UserBg",      "UserFrame"),
    "assistant": ("AssistantBg", "AssistantFrame"),
    "output":    ("OutputBg",    "OutputFrame"),
}


def render_message(role: str, content: str, call_idx: int, msg_idx: int) -> str:
    bg, frame = _ROLE_STYLE.get(role, ("white", "black"))
    title = (
        f"{role.capitalize()}"
        #f"\\textnormal{{\\scriptsize\\ (call\\,{call_idx + 1},\\,msg\\,{msg_idx + 1})}}"
    )

    body_parts: list[str] = []
    for sec in parse_dspy_content(content):
        field = sec["field"]
        text  = sec["content"]

        if field == "completed":
            body_parts.append(_field_header(field))
            rendered = _render_text(text)
            if rendered:
                body_parts.append(rendered)
            body_parts.append("\\smallskip")
            continue

        if field:
            body_parts.append(_field_header(field))

        if _is_code(field, text):
            code_part, trailing = _split_code_trailing(text)
            body_parts.append(_render_code(code_part))
            if trailing:
                rendered = _render_text(trailing)
                if rendered:
                    body_parts.append(rendered)
        else:
            parsed = _try_parse_json(text)
            if parsed is not None:
                body_parts.append(_json_to_latex(parsed, depth=0))
            else:
                rendered = _render_text(text)
                if rendered:
                    body_parts.append(rendered)

        body_parts.append("\\smallskip")

    body = "\n".join(body_parts)

    return (
        "\\begin{tcolorbox}[\n"
        "  breakable, enhanced,\n"
        f"  title={{{title}}},\n"
        f"  colback={bg}, colframe={frame},\n"
        "  coltitle=white, fonttitle=\\bfseries\\small,\n"
        "  arc=2mm, boxrule=1pt,\n"
        "  before upper={\\small\\setlength{\\parindent}{0pt}\\setlength{\\parskip}{2pt}},\n"
        "]\n"
        f"{body}\n"
        "\\end{tcolorbox}\n"
        "\\vspace{0.35em}\n"
    )


# --- Preamble (to be \input-ted in the main document's preamble) ---

_LATEX_PREAMBLE = r"""%% Font encoding: T1 makes \textquotedbl and other symbols available
\usepackage[T1]{fontenc}
%% upquote renders straight quotes in lstlisting (avoids \textquotedbl in OT1)
\usepackage{upquote}

%% tcolorbox with listings support
\usepackage{tcolorbox}
\tcbuselibrary{skins, breakable, listings}

%% Code listings
\usepackage{listings}
\usepackage{xcolor}

%% -- Colour palette ----------------------------------------------------------
\definecolor{SystemBg}      {HTML}{CDDAF5}
\definecolor{SystemFrame}   {HTML}{7799e4}

\definecolor{UserBg}        {HTML}{FFF3CD}
\definecolor{UserFrame}     {HTML}{ffdb66}

\definecolor{AssistantBg}   {HTML}{D4EDDA}
\definecolor{AssistantFrame}{HTML}{8ccf9c}

\definecolor{OutputBg}      {HTML}{FAC898}
\definecolor{OutputFrame}   {HTML}{f69337}

\definecolor{codebg}        {RGB}{248, 248, 248}
\definecolor{codeframe}     {RGB}{180, 180, 180}

\definecolor{kwcolor}       {RGB}{  0,   0, 195}
\definecolor{strcolor}      {RGB}{175,   0,   0}
\definecolor{cmcolor}       {RGB}{ 90, 120,  90}

%% -- Python listing style ----------------------------------------------------
\lstdefinestyle{pythonstyle}{
  language=Python,
  basicstyle=\scriptsize\ttfamily,
  keywordstyle=\color{kwcolor}\bfseries,
  stringstyle=\color{strcolor},
  commentstyle=\color{cmcolor}\itshape,
  numberstyle=\tiny\color{gray},
  numbers=left,
  numbersep=7pt,
  backgroundcolor=\color{codebg},
  frame=single,
  framesep=4pt,
  rulecolor=\color{codeframe},
  breaklines=true,
  breakatwhitespace=false,
  breakautoindent=true,
  breakindent=0.5em,
  postbreak=\mbox{\textcolor{gray}{$\hookrightarrow$}\space},
  keepspaces=true,
  showspaces=false,
  showstringspaces=false,
  tabsize=2,
  xleftmargin=16pt,
  numbersep=8pt,
  xrightmargin=4pt,
  morekeywords={self,True,False,None,as,with,yield,lambda,assert,
                del,except,finally,global,nonlocal,pass,raise,
                return,try,while,async,await},
}
\lstset{style=pythonstyle}
"""


# --- Entry point ---

def _render_history(data: list, parts: list[str]) -> None:
    """Append rendered tcolorboxes for all calls in a history list into parts."""
    for call_idx, call in enumerate(data):
        messages = call.get("messages", [])
        if messages:
            parts.append("\\noindent\\textbf{Prompt}\\par\\smallskip\n")
            for msg_idx, msg in enumerate(messages):
                parts.append(render_message(
                    msg.get("role", "unknown"),
                    msg.get("content", ""),
                    call_idx,
                    msg_idx,
                ))

        outputs = call.get("outputs", [])
        if outputs:
            parts.append("\\noindent\\textbf{Response}\\par\\smallskip\n")
            for out_idx, output in enumerate(outputs):
                parts.append(render_message("output", output, call_idx, out_idx))


def _load_json(path: Path) -> list:
    with open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def generate_from_file(json_path: Path) -> str:
    parts: list[str] = []
    _render_history(_load_json(json_path), parts)
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX from llm_history_propagation.json"
    )
    parser.add_argument("particle_folder", help="Path to the particle folder")
    args = parser.parse_args()

    folder = Path(args.particle_folder).resolve()
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    prop_path = folder / "llm_history_propagation.json"
    fb_path   = folder / "llm_history_feedback.json"

    if not prop_path.exists() and not fb_path.exists():
        print(f"Error: no LLM history files found in {folder}", file=sys.stderr)
        sys.exit(1)

    if prop_path.exists():
        out = folder / "llm_history_propagation.tex"
        out.write_text(generate_from_file(prop_path), encoding="utf-8")
        print(f"Propagation history written to {out}")

    if fb_path.exists():
        out = folder / "llm_history_feedback.tex"
        out.write_text(generate_from_file(fb_path), encoding="utf-8")
        print(f"Feedback history written to {out}")

    preamble_out = folder / "llm_history_preamble.tex"
    preamble_out.write_text(_LATEX_PREAMBLE, encoding="utf-8")
    print(f"Preamble written to {preamble_out}")


if __name__ == "__main__":
    main()
