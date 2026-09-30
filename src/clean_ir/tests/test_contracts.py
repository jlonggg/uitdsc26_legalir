import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
import pipeline
from corpus import build
from package import validate


class Contracts(unittest.TestCase):
    def test_inference_drops_answers_and_accepts_private_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'questions.json'
            p.write_text(json.dumps({'private_A':{'question':'Câu hỏi mới','answer':['secret'],'metadata':'unused'}}))
            self.assertEqual(pipeline.questions(p),{'private_A':{'question':'Câu hỏi mới'}})

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'q.json';p.write_text('{"a":{"question":"x"},"a":{"question":"y"}}')
            with self.assertRaises(ValueError):pipeline.questions(p)

    def test_hash_fold_keeps_normalized_duplicates_together(self):
        self.assertEqual(pipeline.fold('QUY ĐỊNH   về thuế?'),pipeline.fold('quy định về thuế'))

    def test_empty_corpus_documents_are_valid_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);contexts=tmp/'input';contexts.mkdir();out=tmp/'corpus'
            for i,text in [('1','Điều 1. Phạm vi\nNội dung quy định.'),('2','')]:
                (contexts/('context_'+i+'.json')).write_text(json.dumps({'id':i,'name':None,'link':'test','passage':text}))
            build(contexts,out)
            catalog=json.loads((out/'catalog.json').read_text())
            self.assertEqual(set(catalog),{'1','2'})
            self.assertTrue(catalog['2']['empty'])
            validate({'private_A':{}},{'private_A':{'answer':['2']}},catalog)

    def test_membership_and_cardinality_validation(self):
        for a in [['1','1'],['missing'],[],['1','2','3','4','5','6']]:
            with self.assertRaises(ValueError):validate({'q':{}},{'q':{'answer':a}},{str(i):{} for i in range(1,7)})

    def test_article_title_falls_back_to_url_without_page_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);contexts=tmp/'input';contexts.mkdir()
            row={'id':'42','name':None,'link':'https://example.test/Quyet-dinh-ve-thue-123456.aspx',
                 'passage':'Điều 1. Nội dung\nQuy định về thuế.'}
            (contexts/'context_42.json').write_text(json.dumps(row))
            build(contexts,tmp/'corpus')
            article=json.loads((tmp/'corpus/articles.jsonl').read_text().splitlines()[0])
            self.assertTrue(article['header'].startswith('Quyet dinh ve thue'))
            self.assertNotIn('123456',article['header'])

    def test_input_fingerprint_is_content_sensitive(self):
        a={'q':{'question':'old'}};b={'q':{'question':'new'}}
        self.assertNotEqual(pipeline.digest(a),pipeline.digest(b))

    def test_feature_builder_loads_without_train_or_public_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);(tmp/'graph').mkdir()
            (tmp/'doc_meta.json').write_text('{}');(tmp/'doc_nchunks.json').write_text('{}')
            (tmp/'graph/legal_reference_graph.jsonl').write_text('')
            env={'CLEAN_CORPUS':str(tmp),'RB_OUT':str(tmp),'COVFEAT':'1','DETRANK':'1'}
            old={k:os.environ.get(k) for k in env};os.environ.update(env)
            try:
                spec=importlib.util.spec_from_file_location('isolated_features',ROOT/'src/features.py')
                module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                self.assertFalse(hasattr(module,'TRAIN'))
                base=tmp/'ce_public.jsonl';extra=tmp/'ce_xpublic.jsonl'
                base.write_text(json.dumps({'qid':'private_A','docs':[{'doc_id':'1','rank':1,'chunks':['c'],'logits':[1]}]})+'\n')
                extra.write_text(json.dumps({'qid':'private_A','docs':[{'doc_id':'2','chunks':['d'],'logits':[0]},{'doc_id':'2','chunks':['d'],'logits':[0]}]})+'\n')
                self.assertEqual([d['doc_id'] for d in module.load_rows([str(base),str(extra)])['private_A']],['1','2'])
            finally:
                for k,v in old.items():
                    if v is None:os.environ.pop(k,None)
                    else:os.environ[k]=v


if __name__=='__main__':unittest.main()
