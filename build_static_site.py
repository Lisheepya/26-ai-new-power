"""Copy saved experiment artifacts into a build-free static site under docs/."""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent
REPORT_STYLE = """
body{margin:0;background:#f3f6fb;color:#172033;font-family:system-ui,"Noto Sans TC","Microsoft JhengHei",sans-serif;line-height:1.75}
.report-page{max-width:980px;margin:32px auto;padding:32px;background:#fff;border:1px solid #e1e7f0;border-radius:16px;box-shadow:0 14px 40px #1b2f5510}
.report-page h1,.report-page h2,.report-page h3{line-height:1.35}.report-page img{display:block;max-width:100%;height:auto;margin:20px auto;border-radius:8px}
.report-page table{width:100%;border-collapse:collapse;margin:18px 0;display:block;overflow-x:auto}.report-page th,.report-page td{border:1px solid #d8deea;padding:8px 12px;text-align:left}.report-page th{background:#edf2fa}
.report-page pre{overflow:auto;background:#f3f5f9;padding:16px;border-radius:8px}.report-page code{font-family:ui-monospace,Consolas,monospace}.report-page blockquote{border-left:4px solid #9bb2dd;margin:16px 0;padding:4px 16px;color:#52627c;background:#f6f8fc}
.report-nav{max-width:980px;margin:18px auto 0;padding:0 8px}.report-nav a{color:#315fc1}
@media(max-width:700px){.report-page{margin:12px;padding:20px;border-radius:10px}}
"""


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _first_file(root: Path, label: str, candidates: tuple[str, ...]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.is_file():
            return path
    searched = ", ".join(candidates)
    raise FileNotFoundError(f"Missing {label}; checked: {searched}")


def _collect_inputs(root: Path) -> dict[str, Path]:
    return {
        "iris_metrics": _first_file(
            root,
            "Iris metrics JSON",
            ("reports/metrics.json", "reports/iris_metrics.json", "outputs/metrics.json"),
        ),
        "ai4i_metrics": _first_file(
            root,
            "AI4I model metrics JSON",
            (
                "reports/ai4i_modeling/model_comparison.json",
                "reports/model_comparison.json",
                "outputs/ai4i_modeling/model_comparison.json",
            ),
        ),
        "ai4i_eda": _first_file(
            root,
            "AI4I EDA summary JSON",
            ("reports/ai4i_eda/summary.json", "reports/summary.json", "outputs/ai4i_eda/summary.json"),
        ),
        "iris_report": _first_file(
            root,
            "Iris Markdown report",
            ("reports/小論文.md", "reports/iris_report.md", "outputs/小論文.md"),
        ),
        "ai4i_report": _first_file(
            root,
            "AI4I Markdown report",
            ("reports/AI4I_小論文.md", "reports/ai4i_report.md", "outputs/AI4I_小論文.md"),
        ),
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _figure_plan(root: Path) -> list[tuple[Path, Path]]:
    """Return (source, docs-relative destination) pairs for saved PNG files."""
    plan: list[tuple[Path, Path]] = []
    report_figures = root / "reports" / "figures"
    if report_figures.is_dir():
        for source in sorted(report_figures.rglob("*.png")):
            relative = source.relative_to(report_figures)
            plan.append((source, Path("figures") / "reports" / relative))

    outputs = root / "outputs"
    if outputs.is_dir():
        for source in sorted(outputs.rglob("*.png")):
            relative = source.relative_to(outputs)
            if relative.parts[0] in {"ai4i", "ai4i_eda"}:
                destination = Path("figures") / relative
            elif len(relative.parts) == 1:
                destination = Path("figures") / "iris" / relative.name
            else:
                destination = Path("figures") / "outputs" / relative
            plan.append((source, destination))
    return plan


def _resolve_local_link(
    target: str,
    *,
    root: Path,
    report: Path,
    copied_assets: dict[Path, str],
    result_sources: set[Path],
) -> str:
    parsed = urlsplit(target)
    if parsed.scheme or target.startswith(("//", "#")):
        return target
    local_path = Path(unquote(parsed.path))
    candidates = (report.parent / local_path, root / local_path)
    resolved = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
    if resolved in copied_assets:
        return copied_assets[resolved] + (f"#{parsed.fragment}" if parsed.fragment else "")
    if resolved in result_sources:
        return "results.json" + (f"#{parsed.fragment}" if parsed.fragment else "")
    if local_path.suffix.lower() == ".json":
        return "results.json" + (f"#{parsed.fragment}" if parsed.fragment else "")
    return target


def _inline_markdown(
    text: str,
    *,
    root: Path,
    report: Path,
    copied_assets: dict[Path, str],
    result_sources: set[Path],
) -> str:
    escaped = html.escape(text, quote=False)

    def image_replacement(match: re.Match[str]) -> str:
        alt, target = match.groups()
        href = _resolve_local_link(
            html.unescape(target),
            root=root,
            report=report,
            copied_assets=copied_assets,
            result_sources=result_sources,
        )
        return f'<img src="{html.escape(href, quote=True)}" alt="{html.escape(html.unescape(alt), quote=True)}">'

    def link_replacement(match: re.Match[str]) -> str:
        label, target = match.groups()
        href = _resolve_local_link(
            html.unescape(target),
            root=root,
            report=report,
            copied_assets=copied_assets,
            result_sources=result_sources,
        )
        label_html = _inline_markdown(
            html.unescape(label),
            root=root,
            report=report,
            copied_assets=copied_assets,
            result_sources=result_sources,
        )
        return f'<a href="{html.escape(href, quote=True)}">{label_html}</a>'

    escaped = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image_replacement, escaped)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", link_replacement, escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    return escaped


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_table_separator(line: str) -> bool:
    cells = _table_cells(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def markdown_to_html(
    markdown_text: str,
    *,
    title: str,
    root: Path,
    report: Path,
    copied_assets: dict[Path, str],
    result_sources: set[Path],
) -> str:
    """Render the report Markdown subset used by this project using stdlib only."""
    lines = markdown_text.splitlines()
    blocks: list[str] = []
    index = 0

    def inline(value: str) -> str:
        return _inline_markdown(
            value,
            root=root,
            report=report,
            copied_assets=copied_assets,
            result_sources=result_sources,
        )

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("```"):
            language = stripped[3:].strip()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += index < len(lines)
            language_class = f' class="language-{html.escape(language, quote=True)}"' if language else ""
            blocks.append(f"<pre><code{language_class}>{html.escape(chr(10).join(code_lines))}</code></pre>")
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", stripped)
        if heading:
            level = len(heading.group(1))
            blocks.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            index += 1
            continue
        if stripped in {"---", "***", "___"}:
            blocks.append("<hr>")
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines) and _is_table_separator(lines[index + 1].strip()):
            headers = _table_cells(line)
            alignments = [
                "right" if cell.strip().endswith(":") else "left"
                for cell in _table_cells(lines[index + 1])
            ]
            table_rows: list[str] = []
            index += 2
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = _table_cells(lines[index])
                cells += [""] * (len(headers) - len(cells))
                cells = cells[: len(headers)]
                rendered = "".join(
                    f'<td style="text-align:{alignments[column]}">{inline(cell)}</td>'
                    for column, cell in enumerate(cells)
                )
                table_rows.append(f"<tr>{rendered}</tr>")
                index += 1
            header_html = "".join(
                f'<th style="text-align:{alignments[column]}">{inline(cell)}</th>'
                for column, cell in enumerate(headers)
            )
            blocks.append(f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(table_rows)}</tbody></table>")
            continue
        bullet = re.match(r"^\s*[-+*]\s+(.+)$", line)
        ordered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if bullet or ordered:
            is_ordered = ordered is not None
            items: list[str] = []
            pattern = r"^\s*\d+[.)]\s+(.+)$" if is_ordered else r"^\s*[-+*]\s+(.+)$"
            while index < len(lines):
                match = re.match(pattern, lines[index])
                if not match:
                    break
                items.append(f"<li>{inline(match.group(1))}</li>")
                index += 1
            tag = "ol" if is_ordered else "ul"
            blocks.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue
        if stripped.startswith(">"):
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote_lines.append(re.sub(r"^\s*>\s?", "", lines[index]))
                index += 1
            blocks.append(f"<blockquote><p>{inline(' '.join(quote_lines))}</p></blockquote>")
            continue

        paragraph = [stripped]
        index += 1
        while index < len(lines) and lines[index].strip():
            next_line = lines[index].strip()
            if (
                next_line.startswith(("#", "```", ">", "|"))
                or re.match(r"^\s*(?:[-+*]\s+|\d+[.)]\s+)", lines[index])
                or next_line in {"---", "***", "___"}
            ):
                break
            paragraph.append(next_line)
            index += 1
        blocks.append(f"<p>{inline(' '.join(paragraph))}</p>")

    page_title = html.escape(title)
    content = "\n".join(blocks)
    return (
        "<!doctype html>\n<html lang=\"zh-Hant\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>{page_title}</title><style>{REPORT_STYLE}</style></head><body>"
        f'<nav class="report-nav"><a href="index.html">← 回到實驗首頁</a></nav>'
        f'<main class="report-page">{content}</main></body></html>\n'
    )


def build_static_site(root_dir: Path | str = ROOT) -> dict[str, Any]:
    root = Path(root_dir).resolve()
    inputs = _collect_inputs(root)
    figure_plan = _figure_plan(root)
    if not figure_plan:
        raise FileNotFoundError("No existing PNG figures found in reports/figures/ or outputs/.")

    # Validate all source JSON before creating or changing docs/.
    source_json = {key: _load_json(path) for key, path in inputs.items() if key.endswith(("metrics", "eda"))}
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    copied_assets: dict[Path, str] = {}
    figure_manifest: dict[str, list[dict[str, str]]] = {"iris": [], "ai4i": [], "ai4i_eda": [], "other": []}
    for source, relative_destination in figure_plan:
        destination = docs / relative_destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destination_url = relative_destination.as_posix()
        copied_assets[source.resolve()] = destination_url
        group = relative_destination.parts[1] if len(relative_destination.parts) > 1 else "other"
        if group not in figure_manifest:
            group = "other"
        figure_manifest[group].append(
            {"path": destination_url, "source": _relative(root, source), "name": source.stem}
        )

    result_source_paths = {inputs[key].resolve() for key in ("iris_metrics", "ai4i_metrics", "ai4i_eda")}
    report_destinations = {"iris_report": "report_iris.html", "ai4i_report": "report_ai4i.html"}
    report_manifest: list[dict[str, str]] = []
    for key, destination_name in report_destinations.items():
        source = inputs[key]
        title = "Iris 分析報告" if key == "iris_report" else "AI4I 分析報告"
        content = markdown_to_html(
            source.read_text(encoding="utf-8"),
            title=title,
            root=root,
            report=source,
            copied_assets=copied_assets,
            result_sources=result_source_paths,
        )
        (docs / destination_name).write_text(content, encoding="utf-8")
        report_manifest.append(
            {"title": title, "path": destination_name, "source": _relative(root, source)}
        )

    results = {
        "source_files": {key: _relative(root, path) for key, path in inputs.items()},
        "iris": source_json["iris_metrics"],
        "ai4i": source_json["ai4i_metrics"],
        "ai4i_eda": source_json["ai4i_eda"],
        "reports": report_manifest,
        "figures": figure_manifest,
    }
    (docs / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "docs": str(docs),
        "figure_count": sum(len(items) for items in figure_manifest.values()),
        "reports": report_manifest,
        "sources": results["source_files"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root containing saved outputs")
    args = parser.parse_args()
    result = build_static_site(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
