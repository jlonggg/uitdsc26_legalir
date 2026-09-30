"""Underthesea segmentation with a pinned, non-pathological regex executor.

The stock Python re executor backtracks exponentially on URLs followed by long
dotted form fields in issued legal documents. Keep the original lexical pattern,
normalizers and CRF; execute the pattern with regex VERSION0 instead. No timeout,
text truncation or machine-speed-dependent fallback is used.
"""
import importlib
from pathlib import Path
import regex

_pattern = None
_model = None

def segment(text):
    global _pattern, _model
    rt = importlib.import_module('underthesea.pipeline.word_tokenize.regex_tokenize')
    if _pattern is None:
        _pattern = regex.compile(rt.regex_patterns_combine,
                                 regex.VERBOSE | regex.UNICODE | regex.VERSION0)
    text = rt.normalize_characters_in_text(text)
    tokens = [rt.extract_match(m)[0] for m in _pattern.finditer(text)]
    tokens = [rt.token_normalize(t, use_character_normalize=True) for t in tokens]
    if not tokens:
        return ''
    if _model is None:
        from underthesea.models.fast_crf_sequence_tagger import FastCRFSequenceTagger
        ws = importlib.import_module('underthesea.pipeline.word_tokenize')
        _model = FastCRFSequenceTagger()
        _model.load(str(Path(ws.__file__).parent/'models/ws_crf_vlsp2013_20230727'))
    tags = _model.predict([[token] for token in tokens])
    output = []
    for tag, token in zip(tags, tokens):
        if tag == 'I-W' and output:
            output[-1] += ' ' + token
        else:
            output.append(token)
    return ' '.join(item.replace(' ', '_') for item in output)
