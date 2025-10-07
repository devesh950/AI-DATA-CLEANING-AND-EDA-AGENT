from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

import textwrap

INPUT_MD = "README.md"
OUTPUT_PDF = "docs/project_summary_for_recruiters.pdf"


def md_to_text(md_text: str) -> str:
    # A simple markdown-to-plain-text converter for headings and lists
    lines = md_text.splitlines()
    out_lines = []
    for line in lines:
        if line.startswith('#'):
            # heading
            line = line.lstrip('#').strip()
            out_lines.append(line.upper())
            out_lines.append('')
        elif line.startswith('- '):
            out_lines.append('• ' + line[2:].strip())
        else:
            out_lines.append(line)
    return '\n'.join(out_lines)


if __name__ == '__main__':
    import os
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

    with open(INPUT_MD, 'r', encoding='utf-8') as f:
        md = f.read()

    text = md_to_text(md)

    doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    normal = styles['Normal']
    heading = ParagraphStyle('Heading', parent=styles['Heading1'], alignment=TA_CENTER)

    story = []
    story.append(Paragraph('Project: AI Data Cleaning & EDA Agent', heading))
    story.append(Spacer(1, 0.2 * inch))

    # Wrap text into paragraphs
    for block in text.split('\n\n'):
        block = block.strip()
        if not block:
            continue
        # short block -> heading-like
        if block.isupper() and len(block) < 60:
            story.append(Paragraph(block, styles['Heading2']))
        else:
            # wrap long lines
            wrapped = '\n'.join(textwrap.wrap(block, width=100))
            story.append(Paragraph(wrapped.replace('\n','<br/>'), normal))
        story.append(Spacer(1, 0.12 * inch))

    doc.build(story)
    print(f"Wrote {OUTPUT_PDF}")