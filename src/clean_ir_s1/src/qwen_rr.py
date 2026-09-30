"""Base model of the S2 MIL scorer: Qwen3-Reranker-0.6B under the same (question, passage) -> scalar-logit contract as the XLM-R cross-encoder.
Logit = lm_head['yes'] - lm_head['no'] at the last position of the model's own prompt (left padding), i.e. the pretrained
reranker decision; no new classification head. Selected automatically when CLEAN_CE points at a `qwen3` config."""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCT = 'Given a Vietnamese legal question, judge whether the legal document passage contains the provision that answers it'
LORA_TARGETS = r'.*\.(q_proj|k_proj|v_proj|o_proj)$'


def is_qwen(model_dir): return AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True).model_type == 'qwen3'


class QwenRR:
    def __init__(self, model_dir, dtype=torch.float16):
        self.tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, padding_side='left')
        self.pre = self.tok(PREFIX, add_special_tokens=False)['input_ids']; self.suf = self.tok(SUFFIX, add_special_tokens=False)['input_ids']
        self.yes, self.no = self.tok.convert_tokens_to_ids('yes'), self.tok.convert_tokens_to_ids('no')
        self.model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, local_files_only=True)

    def encode(self, question, passages, max_len, device='cuda'):
        body = [f'<Instruct>: {INSTRUCT}\n<Query>: {question}\n<Document>: {p}' for p in passages]
        ids = self.tok(body, add_special_tokens=False, truncation=True, max_length=max_len - len(self.pre) - len(self.suf))['input_ids']
        return self.tok.pad({'input_ids': [self.pre + x + self.suf for x in ids]}, padding=True, return_tensors='pt').to(device)

    def logits(self, model, enc):
        """`model` may be PEFT-wrapped (LoRA layers are injected in place, so the inner decoder already uses them)."""
        base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        h = base.model(**enc).last_hidden_state[:, -1]
        w = base.lm_head.weight[[self.no, self.yes]].to(h.dtype)
        lg = h @ w.T
        return lg[:, 1] - lg[:, 0]
