"""Build every corpus-derived input from issued context JSON, without live indexes."""
import collections
import dataclasses
import json
import re
import sys
import unicodedata
from pathlib import Path
from common import LegalDocument, build_chunks, readable_title
from legal_units import parse_legal_units


def metadata(row):
    slug = row.get('name') or (row.get('link') or '').rstrip('/').split('/')[-1]
    slug = re.sub(r'\.(aspx?|html?)$', '', slug)
    slug = re.sub(r'-\d{5,}$', '', slug)
    slug = ''.join(c for c in unicodedata.normalize('NFD', slug) if unicodedata.category(c) != 'Mn').lower()
    number = re.search(r'(\d+)-(\d{4})-', slug)
    kind = next((t for t in ('luat', 'nghi-dinh', 'thong-tu', 'quyet-dinh')
                 if slug.startswith(t) or slug.startswith('bo-'+t)), 'other')
    return {'toks': [t for t in slug.split('-') if t and not t.isdigit()], 'type': kind,
            'amend': int('sua-doi' in slug), 'year': int(number[2]) if number else 0,
            'num': number[1]+'/'+number[2] if number else ''}


def build(contexts, out):
    out.mkdir(parents=True, exist_ok=True)
    documents, meta, count, catalog = [], {}, {}, {}
    article_count = 0
    with (out/'chunks.jsonl').open('w') as chunks_out, (out/'articles.jsonl').open('w') as articles_out:
        for path in sorted(contexts.glob('context_*.json')):
            row = json.loads(path.read_text())
            doc_id = str(row['id'])
            if doc_id in catalog:
                raise ValueError('Duplicate corpus ID '+doc_id)
            passage = row.get('passage') or ''
            catalog[doc_id] = {'name': row.get('name') or '', 'link': row.get('link') or '', 'empty': not passage.strip()}
            meta[doc_id] = metadata(row)
            if not passage.strip():
                continue
            doc = LegalDocument(doc_id, row.get('name') or '', row.get('link') or '', passage.strip())
            chunks = build_chunks([doc], {'chunking': {'target_chars': 1500, 'max_chars': 1800}})
            count[doc_id] = len(chunks)
            for chunk in chunks:
                chunks_out.write(json.dumps(dataclasses.asdict(chunk), ensure_ascii=False)+'\n')
            # Match corpus-only title reconstruction used by the Article channel:
            # source name if present, otherwise URL basename; drop page ID and hyphens.
            raw_title = doc.name or doc.link.rstrip('/').rsplit('/', 1)[-1].rsplit('.', 1)[0]
            title = readable_title(raw_title)
            for unit in parse_legal_units(doc_id, title, doc.passage):
                if unit.level == 'article':
                    articles_out.write(json.dumps({'doc_id': doc_id, 'unit_id': unit.unit_id,
                        'inst': unit.structure_instance, 'article': unit.article, 'header': unit.parent_header,
                        'body': unit.body, 'boilerplate': unit.boilerplate or ''}, ensure_ascii=False)+'\n')
                    article_count += 1
    assert catalog and sum(count.values()) and article_count
    for name, value in [('catalog', catalog), ('doc_meta', meta), ('doc_nchunks', count)]:
        (out/(name+'.json')).write_text(json.dumps(value, ensure_ascii=False))
    result = {'documents': len(catalog), 'indexed_documents': len(count), 'chunks': sum(count.values()), 'articles': article_count}
    (out/'statistics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    build(Path(sys.argv[1]), Path(sys.argv[2]))
