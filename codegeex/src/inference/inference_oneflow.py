#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import time
import copy
import oneflow as flow
import codegeex
from codegeex.oneflow import CodeGeeXModel
from codegeex.tokenizer import CodeGeeXTokenizer
from codegeex.quantization import quantize_oneflow
from codegeex.data.data_utils import LANGUAGE_TAG
from codegeex.megatron.inference import set_random_seed
from codegeex.oneflow.inference import get_token_stream
os.environ["ONEFLOW_KERNEL_ENABLE_FUSED_LINEAR"] = "1"
os.environ["ONEFLOW_LINEAR_EMBEDDING_SKIP_INIT"] = "1"

model = None
tokenizer = None


def model_provider(args):
    """Build the model."""

    model = CodeGeeXModel(
        args.hidden_size,
        args.num_layers,
        args.num_attention_heads,
        args.padded_vocab_size,
        args.max_position_embeddings
    )
    return model


def add_code_generation_args(parser):
    group = parser.add_argument_group(title="code generation")
    group.add_argument(
        "--num-layers",
        type=int,
        default=39,
    )
    group.add_argument(
        "--hidden-size",
        type=int,
        default=5120,
    )
    group.add_argument(
        "--num-attention-heads",
        type=int,
        default=40,
    )
    group.add_argument(
        "--padded-vocab-size",
        type=int,
        default=52224,
    )
    group.add_argument(
        "--max-position-embeddings",
        type=int,
        default=2048,
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    group.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="Use greedy sampling.",
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
        "--out-seq-length",
        type=int,
        default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--tokenizer-path",
        type=str,
        default="./tokenizer",
    )
    group.add_argument(
        "--load",
        type=str,
    )
    group.add_argument(
        "--state-dict-path",
        type=str,
    )
    group.add_argument(
        "--micro-batch-size",
        type=int,
        default=1,
    )
    group.add_argument(
        "--quantize",
        action="store_true",
    )

    return parser


def load(args) -> codegeex.torch.CodeGeeXModel:
    global model, tokenizer
    try:
        print("Loading tokenizer ...")
        tokenizer = CodeGeeXTokenizer(tokenizer_path=args.tokenizer_path, mode="codegeex-13b")

        print("Loading state dict ...")
        state_dict = flow.load(args.load, map_location="cpu")
        state_dict = state_dict["module"]

        print("Building CodeGeeX model ...")
        model = model_provider(args)
        model.load_state_dict(state_dict)
        model.eval()
        model.half()
        if args.quantize:
            model = quantize_oneflow(model, weight_bit_width=8, backend="torch")
        model.cuda()
        flow.cuda.synchronize()
    except Exception:
        raise


def generate(
        model: CodeGeeXModel,
        tokenizer: CodeGeeXTokenizer,
        prompt: str,
        out_seq_length: int,
        seq_length: int = 2048,
        top_k: int = 0,
        top_p: float = 1.0,
        temperature: float = 1.0,
        micro_batch_size: int = 1,
        backend: str = "megatron",
        greedy: bool = False,
        verbose: bool = False,
):
    tokens = tokenizer.encode_code(prompt)
    n_token_prompt = len(tokens)

    # if verbose:
    #    print(f"Current prompt:\n{prompt}")
    #    print("N_token_prompt:", n_token_prompt)

    generated_codes = []
    if backend == "megatron":
        token_stream = get_token_stream(
            model,
            tokenizer,
            seq_length,
            out_seq_length,
            [copy.deepcopy(tokens) for _ in range(micro_batch_size)],
            micro_batch_size=micro_batch_size,
            topk=top_k,
            topp=top_p,
            temperature=temperature,
            greedy=greedy,
        )
        is_finished = [False for _ in range(micro_batch_size)]
        for i, generated in enumerate(token_stream):
            generated_tokens = generated[0]
            for j in range(micro_batch_size):  # --micro-batch-size 1
                if is_finished[j]:
                    continue

                generated_token_numpy = generated_tokens[j].numpy()
                if generated_token_numpy[-1] == tokenizer.eos_token_id or len(generated_tokens[j]) >= out_seq_length:
                    is_finished[j] = True
                    generated_tokens_ = generated_token_numpy.tolist()
                    generated_code = tokenizer.decode_code(generated_tokens_[n_token_prompt:])
                    generated_code = "".join(generated_code)
                    generated_codes.append(generated_code)
                    return generated_codes
                # if all(is_finished):
                #    break
    return generated_codes


def predict(args, prompt, lang, seed=1234):
    global model, tokenizer
    try:
        set_random_seed(seed)

        # 选择编程语言
        if lang.lower() in LANGUAGE_TAG:
            prompt = LANGUAGE_TAG[lang.lower()] + "\n" + prompt

        if model is None or tokenizer is None:
            assert model is not None or tokenizer is not None, 'CodeGeeX model not load, please load it using load()'

        prompt_tokens = tokenizer.encode_code(prompt)
        n_token_prompt = len(prompt_tokens)

        t0 = time.perf_counter()
        generated_code = generate(
            model,
            tokenizer,
            prompt,
            out_seq_length=args.out_seq_length,
            seq_length=args.max_position_embeddings,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            micro_batch_size=args.micro_batch_size,
            backend="megatron",
            verbose=True,
        )
        t1 = time.perf_counter()
        print("Total generation time:", t1 - t0)

        generated_tokens = tokenizer.encode_code("".join(generated_code))
        n_token_generated = len(generated_tokens)

        result = {
            "PromptTokenNum": n_token_prompt,
            "CompletionTokenNum": n_token_generated,
            "Code": [item.replace("<|endoftext|>", "") for item in generated_code]
        }

        return result
    except Exception:
        raise
