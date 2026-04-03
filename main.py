import os
import sys


MODEL_PATH = "/home/bufsgpu/Hugging-Face/models/Qwen3.5-122B-A10B-FP8/"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's request directly and "
    "correctly. Do not continue with extra questions, examples, or lists "
    "unless the user asks for them."
)


def get_visible_gpu_count() -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_visible_devices or cuda_visible_devices == "-1":
        return 0

    devices = [
        device.strip()
        for device in cuda_visible_devices.split(",")
        if device.strip()
    ]
    return len(devices)


def build_sampling_params():
    from vllm import SamplingParams

    stop_sequences = [
        stop
        for stop in os.getenv("STOP_SEQUENCES", "").split("||")
        if stop
    ]

    return SamplingParams(
        temperature=float(os.getenv("TEMPERATURE", "1.0")),
        top_p=float(os.getenv("TOP_P", "0.95")),
        max_tokens=int(os.getenv("MAX_TOKENS", "81920")),
        repetition_penalty=float(os.getenv("REPETITION_PENALTY", "1.5")),
        stop=stop_sequences or None,
    )


def generate_once(llm, sampling_params, system_prompt: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    response = llm.chat(messages, sampling_params=sampling_params, use_tqdm=False)
    return response[0].outputs[0].text.strip()


def interactive_loop(llm, sampling_params, system_prompt: str) -> None:
    print("Model ready. Enter a prompt. Use /exit to quit.")

    while True:
        try:
            prompt = input("\n>>> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break

        if not prompt:
            continue

        if prompt.lower() in {"/exit", "/quit"}:
            break

        try:
            answer = generate_once(llm, sampling_params, system_prompt, prompt)
        except KeyboardInterrupt:
            print("\nGeneration interrupted.")
            continue

        print(f"\n{answer}")


def main() -> None:
    visible_devices = os.getenv("VISIBLE_GPUS", "0,1")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

    visible_gpus = get_visible_gpu_count()
    tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "2"))
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    gdn_prefill_backend = os.getenv("GDN_PREFILL_BACKEND", "triton")
    system_prompt = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    if visible_gpus == 0:
        raise RuntimeError("No CUDA GPUs are visible to this process.")

    if tensor_parallel_size > visible_gpus:
        raise RuntimeError(
            f"TENSOR_PARALLEL_SIZE={tensor_parallel_size}, but only "
            f"{visible_gpus} GPU(s) are visible. "
            "Check CUDA_VISIBLE_DEVICES or lower TENSOR_PARALLEL_SIZE."
        )

    print(
        f"Using {visible_gpus} visible GPU(s) with "
        f"tensor_parallel_size={tensor_parallel_size}, "
        f"gpu_memory_utilization={gpu_memory_utilization}, "
        f"gdn_prefill_backend={gdn_prefill_backend}"
    )

    from vllm import LLM

    llm = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        gdn_prefill_backend=gdn_prefill_backend,
    )

    sampling_params = build_sampling_params()

    if sys.stdin.isatty():
        interactive_loop(llm, sampling_params, system_prompt)
        return

    prompt = sys.stdin.read().strip()
    if prompt:
        print(generate_once(llm, sampling_params, system_prompt, prompt))


if __name__ == "__main__":
    main()
