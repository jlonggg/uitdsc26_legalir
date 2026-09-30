"""Build dedicated ES chunk/wordseg indexes. Refuses to adopt existing indexes."""
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from itertools import islice
from elasticsearch import Elasticsearch, helpers
from common import lexical_text, title_text
from segmentation import segment


def prepare_chunk(c):
    s = {k: c[k] for k in ('doc_id', 'name', 'article', 'order', 'text')}
    s['lexical'] = lexical_text(f"{c['name']} {c['article']} {c['text']}")
    s['title'] = title_text(c['name'])
    raw = ' '.join(str(s[k]) for k in ('name', 'article', 'title', 'text') if s[k])
    return c['chunk_id'], s, {'doc_id': c['doc_id'], 'wseg': segment(raw)}


def main():
    es = Elasticsearch(os.environ['CLEAN_ES'], request_timeout=180)
    index, wseg = os.environ['CLEAN_INDEX'], os.environ['CLEAN_WSEG']
    owner = os.environ['CLEAN_OWNER']
    for name in (index, wseg):
        if es.indices.exists(index=name):
            mapping = es.indices.get_mapping(index=name)[name]['mappings']
            if mapping.get('_meta', {}).get('clean_ir_owner') != owner:
                raise RuntimeError('Refusing an index not owned by this build: '+name)
        else:
            props = {'doc_id': {'type': 'keyword'}}
            if name == wseg:
                props['wseg'] = {'type': 'text'}
            else:
                props.update({'name': {'type': 'text'}, 'article': {'type': 'keyword'},
                    'order': {'type': 'integer'}, 'lexical': {'type': 'text', 'analyzer': 'whitespace'},
                    'title': {'type': 'text', 'analyzer': 'whitespace'}, 'text': {'type': 'text', 'index': False}})
            es.indices.create(index=name, body={'settings': {'number_of_shards': 1, 'number_of_replicas': 0},
                'mappings': {'dynamic': 'strict', '_meta': {'clean_ir_owner': owner}, 'properties': props}})
    def actions():
        # Bounded batches keep memory independent of corpus size. map preserves
        # input order; worker count does not change the index contents.
        with ProcessPoolExecutor(max_workers=int(os.environ.get('CLEAN_THREADS','3'))) as pool, \
                open(Path(os.environ['CLEAN_CORPUS'])/'chunks.jsonl') as handle:
            while True:
                batch = [json.loads(line) for line in islice(handle, 256)]
                if not batch:
                    break
                for chunk_id, s, segmented in pool.map(prepare_chunk, batch, chunksize=16):
                    yield {'_index': index, '_id': chunk_id, '_source': s}
                    yield {'_index': wseg, '_id': chunk_id, '_source': segmented}
    n = 0
    for success, response in helpers.streaming_bulk(es, actions(), chunk_size=128, request_timeout=180):
        assert success, response
        n += 1
        if n % 10000 == 0:
            print('indexed', n//2, flush=True)
    expected = json.loads((Path(os.environ['CLEAN_CORPUS'])/'statistics.json').read_text())['chunks']
    for name in (index, wseg):
        es.indices.refresh(index=name)
        assert es.count(index=name)['count'] == expected
    (Path(os.environ['CLEAN_CORPUS'])/'index.json').write_text(json.dumps({'owner': owner, 'chunks': expected}))


if __name__ == '__main__':
    main()
