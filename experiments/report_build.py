"""Build report/REPORT.pdf from report/REPORT.md: Markdown -> HTML (python-markdown) -> PDF (headless Chrome).

    python experiments/report_build.py            # writes report/REPORT.html and report/REPORT.pdf
"""
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1] / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 20mm 18mm 20mm 18mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 10.5pt; line-height: 1.4; color: #111;
       max-width: 100%; }
h1 { font-size: 19pt; margin: 0 0 6pt 0; line-height: 1.25; }
h2 { font-size: 14pt; margin: 18pt 0 6pt 0; border-bottom: 1px solid #999; padding-bottom: 2pt; page-break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt 0; page-break-after: avoid; }
p { margin: 4pt 0 6pt 0; text-align: justify; }
ul, ol { margin: 2pt 0 6pt 0; padding-left: 18pt; }
li { margin: 1.5pt 0; }
li > ul { margin-top: 1pt; }
table { border-collapse: collapse; margin: 6pt auto 8pt auto; font-size: 9pt; page-break-inside: auto; }
tr { page-break-inside: avoid; }
thead { display: table-header-group; }
th, td { border: 1px solid #bbb; padding: 2.5pt 6pt; text-align: left; vertical-align: top; }
th { background: #eee; }
tr:nth-child(even) td { background: #f7f7f7; }
code { font-family: Menlo, Consolas, monospace; font-size: 8.8pt; background: #f2f2f2; padding: 0 2pt; }
pre { background: #f2f2f2; padding: 6pt; font-size: 8.5pt; }
figure { margin: 8pt auto 10pt auto; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 95mm; }
figcaption { font-size: 9pt; color: #333; margin-top: 3pt; text-align: center; }
strong { font-weight: 600; }
"""


def to_html(md_text: str) -> str:
    html = markdown.markdown(md_text, extensions=["tables", "sane_lists", "smarty"])
    # images on their own paragraph -> figure with the alt text as caption
    html = re.sub(r'<p><img alt="([^"]*)" src="([^"]*)"\s*/?></p>',
                  r'<figure><img src="\2" alt="\1"><figcaption>\1</figcaption></figure>', html)
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>TrafficFlowBench 2026 report</title>" \
           f"<style>{CSS}</style></head><body>{html}</body></html>"


def main() -> None:
    md_path, html_path, pdf_path = ROOT / "REPORT.md", ROOT / "REPORT.html", ROOT / "REPORT.pdf"
    html_path.write_text(to_html(md_path.read_text()))
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=15000", f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if not pdf_path.exists():
        sys.exit(f"Chrome did not write the PDF:\n{r.stderr[-2000:]}")
    print(f"wrote {html_path} and {pdf_path} ({pdf_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
