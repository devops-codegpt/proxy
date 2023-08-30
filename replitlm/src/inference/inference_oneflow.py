#!/usr/bin/python
# -*- coding: utf-8 -*-
import time

import oneflow as torch
from transformers import AutoTokenizer, AutoConfig
from .langtag import LANGUAGE_TAG

from replitlm.oneflow.modeling_mpt import MPTForCausalLM


def add_code_generation_args(parser):
    group = parser.add_argument_group(title="code generation")
    group.add_argument(
        "--checkpoint",
        type=str,
    )

    group.add_argument(
        "--out-seq-length",
        type=int,
        default=2048,
        help="Size of the output generated text.",
    )

    group.add_argument(
        "--do-sample",
        action="store_true",
        default=False,
        help="Use sampling.",
    )

    group.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top p sampling.",
    )

    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top k sampling.",
    )

    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )

    group.add_argument(
        "--num-return-sequences",
        type=int,
        default=1,
        help="The number of sequence candidates to return for each input.",
    )

    group.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computing device.",
    )

    return parser


def load(args):
    global model, tokenizer
    try:
        print("Loading tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)

        print("Loading model config ...")
        config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
        # config.attn_config["attn_impl"] = "triton"

        print("Loading state dict ...")
        model_file = f"{args.checkpoint.rstrip('/')}/pytorch_model.bin"
        state_dict = torch.load(model_file, map_location="cpu")

        print("Building CodeGeeX model ...")
        model = MPTForCausalLM(config)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.half()
        model.to(device=args.device)
    except Exception:
        raise


def predict(args, prompt, lang):
    global model, tokenizer
    try:
        # 选择编程语言
        if lang.lower() in LANGUAGE_TAG:
            prompt = LANGUAGE_TAG[lang.lower()] + "\n" + prompt

        if model is None or tokenizer is None:
            assert model is not None or tokenizer is not None, 'ReplitLM model not load, please load it using load()'

        prompt_tokens = tokenizer.encode(prompt)
        prompt_tokens = torch.cuda.LongTensor([prompt_tokens])
        n_token_prompt = len(prompt_tokens[0])

        t0 = time.perf_counter()
        generated_tokens = model.generate(
            prompt_tokens,
            max_length=args.out_seq_length,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id
        )
        t1 = time.perf_counter()
        print("Total generation time:", t1 - t0)

        n_token_generated = len(generated_tokens[0][n_token_prompt:])
        result = {
            "PromptTokenNum": n_token_prompt,
            "CompletionTokenNum": n_token_generated,
            "Code": tokenizer.batch_decode(
                generated_tokens[:, n_token_prompt:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        }
        return result
    except Exception:
        raise
