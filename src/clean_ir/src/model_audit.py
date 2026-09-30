"""Count actual loaded parameter tensors for the complete inference DAG."""
import json
import os
import argparse
from pathlib import Path
from transformers import AutoModel, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--adapter');a=ap.parse_args()
    root=Path(os.environ['CLEAN_MODELS']);counts={}
    for name in ('Qwen3-Embedding-0.6B','Vietnamese_Embedding_v2','vietlegal-harrier-0.6b','Vietnamese_Reranker'):
        path=str(root/name)
        if name=='Vietnamese_Reranker':model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True)
        elif name=='Qwen3-Embedding-0.6B':model=AutoModel.from_pretrained(path,local_files_only=True)
        else:model=SentenceTransformer(path,device='cpu',local_files_only=True,trust_remote_code=False)
        counts[name]=sum(p.numel() for p in model.parameters());del model
    # Count the CE student as a separate full backbone; PEFT adapters counted after fit.
    if a.adapter:
        from peft import PeftModel
        base=AutoModelForSequenceClassification.from_pretrained(str(root/'Vietnamese_Reranker'),local_files_only=True)
        student=PeftModel.from_pretrained(base,a.adapter)
        counts['student_with_adapter']=sum(p.numel() for p in student.parameters())
        del student,base
    else:
        counts['student_backbone']=counts['Vietnamese_Reranker']
    total=sum(counts.values())
    assert total<4_000_000_000,total
    out={'actual_loaded_parameters':counts,'total':total,
         'adapter_audit_required_after_fit':not bool(a.adapter),'limit_exclusive':4_000_000_000}
    (Path(os.environ['CLEAN_WORK'])/('parameter_audit_fitted.json' if a.adapter else 'parameter_audit.json')).write_text(json.dumps(out,indent=2))
    print(json.dumps(out),flush=True)


if __name__=='__main__':main()
