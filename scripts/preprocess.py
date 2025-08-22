import re, os, json, pathlib
from pdfminer.high_level import extract_text

RAW = [
    'data/raw/fy24/HDFC_IAR_FY24.pdf',
    'data/raw/fy25/HDFC_IAR_FY25.pdf'
]
CLEAN_DIR = pathlib.Path('data/clean')
SEG_DIR = pathlib.Path('data/segments')
CLEAN_DIR.mkdir(parents=True, exist_ok=True)
SEG_DIR.mkdir(parents=True, exist_ok=True)

HEADINGS = [
    r'CONSOLIDATED\s+BALANCE\s+SHEET',
    r'CONSOLIDATED\s+(STATEMENT\s+OF\s+)?PROFIT\s+AND\s+LOSS',
    r'CONSOLIDATED\s+CASH\s+FLOW',
    r'STANDALONE\s+BALANCE\s+SHEET',
    r'(STATEMENT\s+OF\s+)?PROFIT\s+AND\s+LOSS',
    r'CASH\s+FLOW\s+STATEMENT',
    r'MANAGEMENT\s+DISCUSSION\s+AND\s+ANALYSIS',
    r'NOTES?\s+TO\s+(THE\s+)?ACCOUNTS?',
]

def clean_text(t:str)->str:
    t = t.replace('\f','\n')
    t = re.sub(r'^\s*\d+\s*$', '', t, flags=re.M) # page numbers
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()

def split_on_headings(t:str):
    spans = []
    for m in re.finditer('|'.join(HEADINGS), t, flags=re.I):
        spans.append((m.start(), m.group(0).upper()))
    spans.append((len(t), 'END'))
    if len(spans)<=1:
        return [{'title':'FULL_DOCUMENT','text':t}]
    chunks=[]
    for i in range(len(spans)-1):
        start,title=spans[i]
        end,_=spans[i+1]
        section=t[start:end].strip()
        if section:
            chunks.append({'title':title,'text':section})
    return chunks

def run():
    manifest=[]
    for pdf in RAW:
        p=pathlib.Path(pdf)
        if not p.exists():
            print(f"Missing {pdf}, skipping"); continue
        txt=extract_text(pdf)
        cleaned=clean_text(txt)
        (CLEAN_DIR / (p.stem + '.txt')).write_text(cleaned, encoding='utf-8')
        segs=split_on_headings(cleaned)
        out=(SEG_DIR / (p.stem + '.jsonl'))
        with out.open('w', encoding='utf-8') as f:
            for c in segs:
                f.write(json.dumps({'source': str(p), **c}, ensure_ascii=False)+'\n')
        manifest.append({'pdf': str(p), 'clean': str(CLEAN_DIR / (p.stem + '.txt')), 'segments': str(out), 'sections': len(segs)})
    (SEG_DIR/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Done. See data/segments/manifest.json')

if __name__=='__main__':
    run()
