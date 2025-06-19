import argparse
import asyncio
import time
from dataclasses import dataclass
from importlib.metadata import version
from itertools import chain
from pathlib import Path
from random import choices

import numpy as np
import torch
from PIL import Image
from tensorrt_llm.bench.benchmark.utils.general import get_settings_from_engine
from tensorrt_llm.bench.dataclasses.configuration import RuntimeConfig
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.inputs import PromptInputs, prompt_inputs
from tensorrt_llm.llmapi import LLM, CapacitySchedulerPolicy, RequestOutput, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class Request:
    inputs: PromptInputs
    output_tokens: int


@dataclass
class PerfItem:
    start_timestamp: int
    end_timestamp: int
    token_timestamps: list[int]
    request_id: int
    num_input_tokens: int
    response_is_final: bool
    error: bool
    tokens: list[int]
    decoding_iteration: int
    time_on_first_token: int
    streaming: bool


@dataclass
class BenchmarkMetrics:
    total_duration_s: float
    total_input: int
    total_output: int
    request_throughput: float
    output_throughput: float
    total_token_throughput: float
    mean_ttft_ms: float
    mean_tpots_ms: float
    mean_itl_ms: list[float]
    mean_e2el_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--engine-dir", type=str, default=None)
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "autodeploy", "tensorrt"])
    parser.add_argument("--dataset", type=str, default="random", choices=["random"])
    parser.add_argument("--random-seed", type=int, default=42)

    # runtime options
    parser.add_argument(
        "--max-batch-size", type=int, default=None, help="Maximum runtime batch size to run the engine with"
    )
    parser.add_argument(
        "--max-num-tokens", type=int, default=None, help="Maximum runtime tokens that an engine can accept"
    )
    parser.add_argument("--max-seq-len", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--beam-width", type=int, default=1, help="Number of search beams")
    parser.add_argument(
        "--kv-cache-free-gpu-mem-fraction",
        type=float,
        default=0.90,
        help="The percentage of memory to use for KV cache after model load",
    )
    parser.add_argument(
        "--scheduler-policy",
        type=str,
        default="guaranteed_no_evict",
        choices=["guaranteed_no_evict", "max_utilization", "static_batch"],
    )

    # benchmark options
    parser.add_argument("--num-requests", type=int, default=100, help="Number of requests to cap benchmark run at")
    parser.add_argument("--num-image", type=int, default=1, help="Number of images per request")
    parser.add_argument("--warmup", type=int, default=2, help="Number of requests warm up benchmark")
    parser.add_argument(
        "--input-len", type=int, default=512, help="Target input sequence length (including image tokens)"
    )
    parser.add_argument("--output-len", type=int, default=512, help="Target output sequence length")
    parser.add_argument("--concurrency", type=int, default=-1, help="Desired concurrency rate")
    parser.add_argument("--streaming", action="store_true", default=False, help="Enable streaming mode for requests")
    parser.add_argument("--use-cuda-graph", action="store_true", default=False, help="Use CUDA graph for inference")
    parser.add_argument("--iteration-log", type=str, default=None, help="Path to iteration log file")

    return parser.parse_args()


def get_llm_args(args: argparse.Namespace) -> dict:
    max_seq_len = args.max_seq_len
    model = args.model
    engine_dir = Path(args.engine_dir) if args.engine_dir is not None else None
    backend = args.backend
    iteration_log = None  # not supported

    kwargs = {}
    if backend.lower() in ["pytorch", "autodeploy"]:
        kv_cache_dtype = "auto"
        world_config = {"pp_size": 1, "tp_size": 1, "world_size": 1, "ep_size": 1}
        if args.max_batch_size and args.max_num_tokens:
            max_batch_size, max_num_tokens = args.max_batch_size, args.max_num_tokens
        else:
            max_batch_size, max_num_tokens = BuildConfig.max_batch_size, BuildConfig.max_num_tokens

        # model_config = get_model_config(args.model)
        # trtllm_model_config = ModelConfig.from_pretrained(model, trust_remote_code=True)

        pyt_options = {
            "use_cuda_graph": args.use_cuda_graph,
            "enable_overlap_scheduler": True,
            "kv_cache_dtype": kv_cache_dtype,
        }
        exec_settings = {
            "sw_version": version("tensorrt_llm"),
            "model_path": None,
            "settings_config": {
                "max_batch_size": max_batch_size,
                "max_num_tokens": max_num_tokens,
                "chunking": False,
            },
            "world_config": world_config,
            "backend": backend,
            "decoding_config": {},
            "performance_options": {
                "cuda_graphs": pyt_options["use_cuda_graph"],
                "pytorch_config": pyt_options,
            },
        }
        exec_settings["model"] = model
    else:
        assert engine_dir is not None, "Engine directory is required for C++ backend"
        assert max_seq_len is None, "max_seq_len is not a runtime parameter for C++ backend"
        exec_settings, build_cfg = get_settings_from_engine(engine_dir / "llm")
        kwargs["max_seq_len"] = build_cfg["max_seq_len"]
        exec_settings["model"] = str(engine_dir / "llm")


    # runtime options
    runtime_max_bs = args.max_batch_size or exec_settings["settings_config"]["max_batch_size"]
    runtime_max_tokens = args.max_num_tokens or exec_settings["settings_config"]["max_num_tokens"]
    kv_cache_percent = args.kv_cache_free_gpu_mem_fraction
    beam_width = args.beam_width

    exec_settings["settings_config"]["kv_cache_percent"] = kv_cache_percent
    exec_settings["settings_config"]["max_batch_size"] = runtime_max_bs
    exec_settings["settings_config"]["max_num_tokens"] = runtime_max_tokens
    exec_settings["settings_config"]["beam_width"] = beam_width
    exec_settings["settings_config"]["scheduler_policy"] = CapacitySchedulerPolicy(args.scheduler_policy.upper())

    exec_settings["settings_config"]["dynamic_max_batch_size"] = True
    exec_settings["iteration_log"] = iteration_log

    runtime_config = RuntimeConfig(**exec_settings)
    kwargs = kwargs | runtime_config.get_llm_args()
    kwargs["backend"] = backend

    if "pytorch_backend_config" in kwargs and iteration_log is not None:
        kwargs["pytorch_backend_config"].enable_iter_perf_stats = True

    return kwargs


def load_real_requests(tokenizer: AutoTokenizer, image_size: tuple[int, int] = (504, 504)) -> list[Request]:
    urls = [
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png",
    ]
    from requests import get

    requests: list[Request] = []
    for url in urls:
        image = Image.open(get(url, stream=True).raw).convert("RGB").resize(image_size)
        image = torch.from_numpy(np.array(image).transpose(2, 0, 1) / 255.0).to(torch.float32)
        text = "Question: Describe this image. Answer:"
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image", "image": image},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        requests.append(
            Request(
                inputs=prompt_inputs({"prompt": prompt, "multi_modal_data": {"image": [image]}}),
                output_tokens=128,
            )
        )

    return requests


def load_requests(
    name: str,
    *,
    num_requests: int = 100,
    isl: int = 0,
    osl: int = 0,
    num_image: int = 1,
    image_size: tuple[int, int] = (504, 504),
) -> list[Request]:
    requests: list[Request] = []
    if name == "random":
        assert isl > 0 and osl > 0, "Input and output sequence length must be greater than 0"
        assert isl - 324 * num_image > 0, "Input sequence length must be greater than 324 * num_image"
        for _ in range(num_requests):
            prompt = "<|image_pad|>" * num_image + "pad" * (isl - 324 * num_image)
            images = [torch.rand(3, *image_size) for _ in range(num_image)]

            requests.append(
                Request(
                    inputs=prompt_inputs(
                        {
                            "prompt": prompt,
                            "multi_modal_data": {"image": images},
                        }
                    ),
                    output_tokens=osl,
                )
            )
    else:
        raise ValueError(f"Invalid dataset name: {name}")

    return requests


def calculate_metrics(perf_items: list[PerfItem], dur_s: float) -> BenchmarkMetrics:
    total_input = 0
    total_output = 0
    tpots_ms: list[float] = []
    ttfts_ms: list[float] = []
    itls_ms: list[list[float]] = []
    e2els_ms: list[float] = []

    for perf_item in perf_items:
        input_len = perf_item.num_input_tokens
        output_len = len(perf_item.tokens)
        total_input += input_len
        total_output += output_len
        latency_ms = (perf_item.end_timestamp - perf_item.start_timestamp) / 1e6
        e2els_ms.append(latency_ms)

        if perf_item.streaming:
            ttft_ms = (perf_item.time_on_first_token - perf_item.start_timestamp) / 1e6
            if output_len > 1:
                tpots_ms.append((latency_ms - ttft_ms) / (output_len - 1))
            ttfts_ms.append(ttft_ms)
            itl: list[float] = []
            for i in range(1, len(perf_item.token_timestamps)):
                itl.append((perf_item.token_timestamps[i] - perf_item.token_timestamps[i - 1]) / 1e6)
            itls_ms.append(itl)

    return BenchmarkMetrics(
        total_duration_s=dur_s,
        total_input=total_input,
        total_output=total_output,
        request_throughput=len(perf_items) / dur_s,
        output_throughput=total_output / dur_s,
        total_token_throughput=(total_input + total_output) / dur_s,
        mean_ttft_ms=np.mean(ttfts_ms or 0),
        mean_tpots_ms=np.mean(tpots_ms or 0),
        mean_itl_ms=np.mean(itls_ms or 0),
        mean_e2el_ms=np.mean(e2els_ms or 0),
    )


async def generate_request(
    llm: LLM,
    request: Request,
    sampling_params: SamplingParams,
    streaming: bool,
    pbar: tqdm,
) -> PerfItem:
    sampling_params.max_tokens = request.output_tokens

    request_start_timestamp = time.perf_counter_ns()
    time_on_first_token = None
    output: RequestOutput = llm.generate_async(request.inputs, sampling_params=sampling_params, streaming=streaming)
    token_timestamps: list[int] = []
    if streaming:
        async for stream_output in output:
            if time_on_first_token is None:
                time_on_first_token = time.perf_counter_ns()

            token_timestamps.append(time.perf_counter_ns())

        response = stream_output
    else:
        response: RequestOutput = await output.aresult()

    response_end_timestamp = time.perf_counter_ns()
    pbar.update(1)

    tokens = list(chain(*[beam.token_ids for beam in response.outputs]))
    return PerfItem(
        start_timestamp=request_start_timestamp,
        end_timestamp=response_end_timestamp,
        token_timestamps=token_timestamps,
        request_id=response.request_id,
        num_input_tokens=len(response.prompt_token_ids),
        response_is_final=response.finished,
        error=False,
        tokens=tokens,
        decoding_iteration=response.decoding_iter,
        time_on_first_token=time_on_first_token,
        streaming=streaming,
    )


async def process_request(
    llm: LLM,
    request: Request,
    sampling_params: SamplingParams,
    sem: asyncio.Semaphore | None,
    streaming: bool,
    pbar: tqdm,
) -> PerfItem:
    sampling_params.max_tokens = request.output_tokens

    if sem is None:
        return await generate_request(llm, request, sampling_params, streaming, pbar)
    async with sem:
        return await generate_request(llm, request, sampling_params, streaming, pbar)


async def benchmark(
    llm: LLM,
    *,
    sampling_params: SamplingParams,
    requests: list[Request],
    concurrency: int = -1,
    streaming: bool = False,
) -> BenchmarkMetrics:
    pbar = tqdm(total=len(requests))
    sem = asyncio.Semaphore(concurrency) if concurrency > 0 else None
    tasks: list[asyncio.Task] = []

    benchmark_start_time = time.perf_counter_ns()

    for request in requests:
        tasks.append(asyncio.create_task(process_request(llm, request, sampling_params, sem, streaming, pbar)))

    output_perfs: list[PerfItem] = await asyncio.gather(*tasks)
    benchmark_duration = time.perf_counter_ns() - benchmark_start_time
    pbar.close()

    return calculate_metrics(output_perfs, benchmark_duration / 1e9)


if __name__ == "__main__":
    args = parse_args()
    print("Preparing to run benchmark...")
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    requests = load_requests(
        args.dataset,
        num_requests=args.num_requests,
        isl=args.input_len,
        osl=args.output_len,
        num_image=args.num_image,
    )

    llm_args = get_llm_args(args)
    print(f"LLM arguments: {llm_args}")
    print("Loading model...")
    llm = LLM(**llm_args)

    sampling_params = SamplingParams(
        end_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.all_special_ids[0],
        max_tokens=128,
        ignore_eos=True,
    )

    if args.warmup > 0:
        warmup_requests = choices(requests, k=args.warmup)
        print("Running warmup...")
        asyncio.run(
            benchmark(
                llm,
                sampling_params=sampling_params,
                requests=warmup_requests,
                concurrency=args.concurrency,
                streaming=args.streaming,
            )
        )
        llm._executor._iter_stats_result = None
        print("Warmup done.")

    print("Starting benchmark...")
    metrics = asyncio.run(
        benchmark(
            llm,
            sampling_params=sampling_params,
            requests=requests,
            concurrency=args.concurrency,
            streaming=args.streaming,
        )
    )
    print("Benchmark complete.")

    print(f" Benchmark duration (s): {metrics.total_duration_s:.3f}")
    print(f" Total input tokens: {metrics.total_input}")
    print(f" Total output tokens: {metrics.total_output}")
    print(f" Request throughput (req/s): {metrics.request_throughput:.3f}")
    print(f" Output token throughput (tok/s): {metrics.output_throughput:.3f}")
    print(f" Total token throughput (tok/s): {metrics.total_token_throughput:.3f}")
    print(f" Mean E2E latency (ms): {metrics.mean_e2el_ms:.3f}")
    if args.streaming:
        print(f" Mean TTFT (ms): {metrics.mean_ttft_ms:.3f}")
        print(f" Mean TPOTS (ms): {metrics.mean_tpots_ms:.3f}")
