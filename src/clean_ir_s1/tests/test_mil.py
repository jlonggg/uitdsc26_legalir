import importlib.util, json, os, random, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from mil_common import build_groups, pick, fold


class MilLane(unittest.TestCase):
    def test_lexalign_prefers_copied_phrase(self):
        import lexalign as L
        L.TEXT.update({'a': 'Thời hạn cấp đăng ký xe máy là ba mươi ngày kể từ ngày nộp hồ sơ.', 'b': 'Xe máy phải đăng ký theo quy định về thời hạn.'})
        r = L.align('Thời hạn cấp đăng ký xe máy là bao lâu?', [['a'], ['b'], ['zz']])
        self.assertGreater(r[0][0], r[1][0]); self.assertGreater(r[0][2], r[1][2]); self.assertEqual(r[2], [0.0]*4+[r[2][4], r[2][5]]); self.assertEqual(r[0][4], 0.0)
    def test_pick_dedupes_and_pads_in_order(self):
        self.assertEqual(pick(['a', 'b'], ['b', 'c'], ['a', 'd', 'e'], 4), ['a', 'b', 'c', 'd'])
        self.assertEqual(pick([], [], ['x'], 4), ['x'])

    def test_groups_never_use_gold_or_duplicate_gold_as_negative(self):
        train = {'1': {'question': 'Thuế thu nhập cá nhân?', 'answer': ['g1', 'g2']},
                 '2': {'question': 'thuế thu nhập cá nhân', 'answer': ['g3']},   # normalized duplicate of 1: its gold is banned for 1
                 '3': {'question': 'Câu khác hẳn', 'answer': ['zz']}}
        ev = {q: [{'doc_id': d, 'chunks': [d+'c']} for d in ['g1', 'g2', 'g3', 'n1', 'n2', 'n3', 'n4']] for q in train}
        ranking = {q: ['g3', 'n1', 'g1', 'n2', 'n3', 'n4', 'g2'] for q in train}
        outer = next(k for k in range(5) if all(fold(r['question']) != k for r in train.values()))
        groups, skipped = build_groups(train, ev, ranking, outer, n_hard=2, n_rand=2, rng=random.Random(0))
        by = {}
        for g in groups: by.setdefault(g['qid'], []).append(g)
        self.assertEqual(len(by['1']), 2)                      # one group per gold present in the pool
        for g in by['1']:
            self.assertIn(g['docs'][0], {'g1', 'g2'})          # positive first
            self.assertFalse({'g1', 'g2', 'g3'} & set(g['docs'][1:]))
            self.assertEqual(g['docs'][1:3], ['n1', 'n2'])     # hardest non-banned docs in ranking order
            self.assertEqual(len(g['docs']), 5); self.assertEqual(len(g['chunks']), 5)
        self.assertEqual(skipped, 1)                           # question 3: gold not in pool

    def test_feature_width_per_scorer(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp/'graph').mkdir()
            (tmp/'doc_meta.json').write_text('{}'); (tmp/'doc_nchunks.json').write_text('{}'); (tmp/'graph/legal_reference_graph.jsonl').write_text('')
            (tmp/'best_chunk_text.jsonl').write_text('')
            env = {'CLEAN_CORPUS': str(tmp), 'RB_OUT': str(tmp), 'COVFEAT': '1', 'DETRANK': '1'}
            old = {k: os.environ.get(k) for k in env}; os.environ.update(env)
            try:
                spec = importlib.util.spec_from_file_location('feat_iso', ROOT/'src/features.py'); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
                sel = importlib.util.spec_from_file_location('sel_iso', ROOT/'src/selector_stage.py'); S = importlib.util.module_from_spec(sel)
                sys.modules['features'] = F; sel.loader.exec_module(S)
                docs = [{'doc_id': str(i), 'rank': i+1, 'chunks': ['c%d' % i], 'logits': [float(-i)]} for i in range(6)]
                dense = (['0', '1'], [0.9, 0.8]); extra = F.extra_q([{'q': (['1'], [0.5])}, {'q': (['2'], [0.4])}], 'q')
                st = {d['doc_id']: 0.1 for d in docs}
                self.assertEqual(F.features(docs, dense, student=st, question='q', extra=extra, p3=st).shape[1], S.WIDTH['ce'])
                self.assertEqual(F.features(docs, dense, student=None, question='q', extra=extra, p3=None).shape[1], S.WIDTH['mil'])
                self.assertEqual(F.features(docs, dense, student=st, question='q', extra=extra, p3=st, qrr=st).shape[1], S.WIDTH['ce+mil'])
                self.assertEqual(F.features(docs, dense, student=st, question='q', extra=extra, p3=st, qrr=st, qrr2=st).shape[1], S.WIDTH['ce+mil2'])
                self.assertEqual(F.features(docs, dense, student=st, question='q', extra=extra, p3=st, lex={'0': [0.1]*6}).shape[1], S.WIDTH['ce+lex'])
                self.assertEqual(F.features(docs, dense, student=st, question='q', extra=extra, p3=st, qrr=st, qrr2=st, lex={'0': [0.1]*6}).shape[1], S.WIDTH['ce+mil2+lex'])
            finally:
                for k, v in old.items():
                    if v is None: os.environ.pop(k, None)
                    else: os.environ[k] = v

    def test_mil_rows_accepted_by_pool_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp/'graph').mkdir()
            (tmp/'doc_meta.json').write_text('{}'); (tmp/'doc_nchunks.json').write_text('{}'); (tmp/'graph/legal_reference_graph.jsonl').write_text('')
            env = {'CLEAN_CORPUS': str(tmp), 'RB_OUT': str(tmp), 'COVFEAT': '1', 'DETRANK': '1'}
            old = {k: os.environ.get(k) for k in env}; os.environ.update(env)
            try:
                spec = importlib.util.spec_from_file_location('feat_iso2', ROOT/'src/features.py'); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
                p = tmp/'held_mil.jsonl'; p.write_text(json.dumps({'qid': 'a', 'docs': [{'doc_id': '1', 'rank': 1, 'chunks': ['c'], 'logits': [1.0]}, {'doc_id': '2', 'rank': 61, 'chunks': ['d'], 'logits': [0.0]}]})+'\n')
                self.assertEqual([d['doc_id'] for d in F.load_rows([str(p)])['a']], ['1', '2'])
                with self.assertRaises(SystemExit): F.load_rows([str(tmp/'random_scores.jsonl')])
            finally:
                for k, v in old.items():
                    if v is None: os.environ.pop(k, None)
                    else: os.environ[k] = v


if __name__ == '__main__':
    unittest.main()
