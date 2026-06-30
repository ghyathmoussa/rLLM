#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

def send_request(url, model, max_tokens, stream):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a short paragraph explaining the benefits of Rust for systems programming."}],
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.7
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    
    start_time = time.time()
    ttft = None
    tokens_generated = 0
    
    try:
        # 30-second timeout per request
        with urllib.request.urlopen(req, timeout=30) as response:
            if stream:
                for line in response:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        if ttft is None:
                            ttft = time.time() - start_time
                        data_content = line[5:].strip()
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_content)
                            if chunk.get("choices") and chunk["choices"][0].get("delta", {}).get("content"):
                                tokens_generated += 1
                        except Exception:
                            pass
            else:
                resp_data = response.read().decode('utf-8')
                end_time = time.time()
                resp_json = json.loads(resp_data)
                ttft = end_time - start_time
                if "usage" in resp_json:
                    tokens_generated = resp_json["usage"]["completion_tokens"]
                else:
                    tokens_generated = max_tokens
                    
        end_time = time.time()
        latency = end_time - start_time
        return {
            "success": True,
            "latency": latency,
            "ttft": ttft,
            "tokens": tokens_generated,
            "error": None
        }
    except urllib.error.HTTPError as e:
        error_msg = f"HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')}"
        return {
            "success": False,
            "latency": time.time() - start_time,
            "ttft": None,
            "tokens": 0,
            "error": error_msg
        }
    except Exception as e:
        return {
            "success": False,
            "latency": time.time() - start_time,
            "ttft": None,
            "tokens": 0,
            "error": str(e)
        }

def run_benchmark(args):
    print("=" * 60)
    print("rLLM Concurrency Test Benchmark")
    print(f"Server URL:     {args.url}")
    print(f"Model ID:       {args.model}")
    print(f"Concurrency:    {args.concurrency}")
    print(f"Total Requests: {args.total_requests}")
    print(f"Max Tokens:     {args.max_tokens}")
    print(f"Streaming mode: {'Enabled' if args.stream else 'Disabled'}")
    print("=" * 60)
    print("Warming up server and submitting requests...")
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(send_request, args.url, args.model, args.max_tokens, args.stream): i
            for i in range(args.total_requests)
        }
        
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            completed += 1
            if completed % max(1, args.total_requests // 10) == 0 or completed == args.total_requests:
                print(f" Progress: {completed}/{args.total_requests} requests finished...")
                
    elapsed = time.time() - start_time
    
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    print(f"Successful requests: {len(successes)} / {args.total_requests} ({len(successes)/args.total_requests*100:.1f}%)")
    print(f"Failed requests:     {len(failures)} / {args.total_requests}")
    
    if failures:
        print("\nErrors encountered:")
        errors = set(f["error"] for f in failures)
        for err in list(errors)[:5]:
            print(f" - {err}")
            
    if successes:
        latencies = [s["latency"] for s in successes]
        ttfts = [s["ttft"] for s in successes if s["ttft"] is not None]
        total_tokens = sum(s["tokens"] for s in successes)
        
        avg_latency = sum(latencies) / len(latencies)
        p50_latency = sorted(latencies)[int(len(latencies) * 0.50)]
        p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
        
        print(f"\nThroughput:")
        print(f"  Request Throughput: {len(successes)/elapsed:.2f} requests/sec")
        print(f"  Token Throughput:   {total_tokens/elapsed:.2f} tokens/sec")
        
        print(f"\nLatency metrics:")
        print(f"  Average End-to-End Latency: {avg_latency:.3f} seconds")
        print(f"  P50 End-to-End Latency:      {p50_latency:.3f} seconds")
        print(f"  P95 End-to-End Latency:      {p95_latency:.3f} seconds")
        
        if ttfts:
            avg_ttft = sum(ttfts) / len(ttfts)
            p50_ttft = sorted(ttfts)[int(len(ttfts) * 0.50)]
            p95_ttft = sorted(ttfts)[int(len(ttfts) * 0.95)]
            print(f"  Average TTFT:               {avg_ttft:.3f} seconds")
            print(f"  P50 TTFT:                   {p50_ttft:.3f} seconds")
            print(f"  P95 TTFT:                   {p95_ttft:.3f} seconds")
            
        print(f"  Total tokens generated:     {total_tokens}")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="rLLM Concurrency Test Load Generator")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="rLLM server base URL")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model identifier to specify in payload")
    parser.add_argument("--concurrency", type=int, default=10, help="Number of concurrent workers/threads")
    parser.add_argument("--total-requests", type=int, default=50, help="Total number of requests to execute")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max tokens to generate per request")
    parser.add_argument("--stream", action="store_true", help="Enable SSE streaming to compute true TTFT")
    
    args = parser.parse_args()
    run_benchmark(args)
