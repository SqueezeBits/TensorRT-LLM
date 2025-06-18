import argparse

import torch
from tensorrt_llm.llmapi import LLM, SamplingParams
from transformers import AutoTokenizer


BACKENDS = [
    "trt",
    "pytorch",
]

MODELS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", choices=MODELS)
    parser.add_argument("--num-image", type=int, default=2)
    parser.add_argument("--input-seqlen", type=int, default=1024)
    parser.add_argument("--output-seqlen", type=int, default=512)
    parser.add_argument("--image-height", type=int, default=504)
    parser.add_argument("--image-width", type=int, default=504)
    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sampling_params = SamplingParams(max_tokens=args.output_seqlen, ignore_eos=True)
    prompt = "<|image_pad|>" * args.num_image + "pad" * (args.input_seqlen - 324*args.num_image)
    multi_modal_data = {
        "image": [torch.rand(3, args.image_height, args.image_width).to("cuda"),] * args.num_image
    }
    inputs = [{
        "prompt": prompt,
        "multi_modal_data": multi_modal_data
    }]
    with LLM("/code/tensorrt_llm/engines/qwen2.5-vl/llm", tokenizer=tokenizer) as llm:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        for i, output in enumerate(outputs):
            print(f"[{i}] Context length: {len(output.prompt_token_ids)}, Generation length: {len(output.outputs[0].token_ids)}")


if __name__ == "__main__":
    main()
