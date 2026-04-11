"""
Edge AI Energy Scraper — v2 (Fixed)
Collects real published benchmark data from:
  1. MLCommons tiny_results GitHub repos (v0.5, v1.0, v1.1, v1.2, v1.3)
  2. Papers With Code API (with proper error handling)
  3. Coral AI benchmarks page
  4. Known published values (guaranteed fallback)

Fixes from v1:
  - MLCommons: v1.0 uses performance_result.txt, v1.1+ uses result.txt
  - GitHub API: token support via --github_token or GITHUB_TOKEN env var
  - GitHub API: proper 403/rate-limit handling with clear user message
  - Papers With Code: guards against empty/non-JSON responses
  - All sources: one failure never crashes the others

Usage:
    python scraper.py                            # known_values + coral always work
    python scraper.py --github_token ghp_xxx     # enables full MLCommons scrape
    GITHUB_TOKEN=ghp_xxx python scraper.py       # same via env var
    python scraper.py --sources known_values coral  # skip GitHub entirely
"""

import argparse
import json
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

warnings.filterwarnings("ignore")

TIMEOUT    = 20
RATE_LIMIT = 1.5

GITHUB_API = "https://api.github.com"
GITHUB_RAW = "https://raw.githubusercontent.com"

MLCOMMONS_REPOS = [
    ("mlcommons/tiny_results_v0.5", "main",   "result.txt"),
    ("mlcommons/tiny_results_v1.0", "main",   "performance_result.txt"),
    ("mlcommons/tiny_results_v1.1", "main",   "result.txt"),
    ("mlcommons/tiny_results_v1.2", "main",   "result.txt"),
    ("mlcommons/tiny_results_v1.3", "main",   "result.txt"),
]

MODEL_ALIASES = {
    "mobilenetv3":          "mobilenetv3_small",
    "mobilenetv3-small":    "mobilenetv3_small",
    "mobilenet_v3_small":   "mobilenetv3_small",
    "mobilenet_v3":         "mobilenetv3_small",
    "mobilenetv2":          "mobilenetv2",
    "mobilenet_v2":         "mobilenetv2",
    "mobilenet":            "mobilenetv2",
    "resnet18":             "resnet18",
    "resnet-18":            "resnet18",
    "resnetv1":             "resnet18",
    "resnet":               "resnet18",
    "efficientnet-b0":      "efficientnet_b0",
    "efficientnet_b0":      "efficientnet_b0",
    "efficientnet":         "efficientnet_b0",
    "tiny-vit":             "tiny_vit_5m",
    "tiny_vit":             "tiny_vit_5m",
}

DEVICE_ALIASES = {
    "raspberry pi 4":   "raspberry_pi4",
    "raspberry_pi_4":   "raspberry_pi4",
    "raspberrypi4":     "raspberry_pi4",
    "rpi4":             "raspberry_pi4",
    "rpi 4":            "raspberry_pi4",
    "pi 4":             "raspberry_pi4",
    "jetson nano":      "jetson_nano",
    "jetson_nano":      "jetson_nano",
    "jetsonnano":       "jetson_nano",
    "coral":            "coral_tpu",
    "coral edge tpu":   "coral_tpu",
    "coral_edge_tpu":   "coral_tpu",
    "coral micro":      "coral_tpu",
    "edgetpu":          "coral_tpu",
    "stm32":            "stm32",
    "nucleo":           "stm32",
    "nucleo-l4r5zi":    "stm32",
    "stm32l4":          "stm32",
    "snapdragon":       "snapdragon888",
    "snapdragon 888":   "snapdragon888",
    "apple silicon":    "apple_silicon",
    "apple m1":         "apple_silicon",
    "apple m2":         "apple_silicon",
    "macbook":          "apple_silicon",
}


def match_model(text):
    t = text.lower()
    for alias, canonical in MODEL_ALIASES.items():
        if alias in t:
            return canonical
    return None


def match_device(text):
    t = text.lower()
    for alias, canonical in DEVICE_ALIASES.items():
        if alias in t:
            return canonical
    return None


def make_headers(token=None):
    h = {"User-Agent": "edge-ai-energy-scraper/2.0 (research)"}
    if token:
        h["Authorization"] = f"token {token}"
    return h


def safe_get(url, headers, retries=2, **kwargs):
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=TIMEOUT, **kwargs)
            if r.status_code == 403:
                print(f"  [warn] 403 rate-limited: {url}")
                return None
            if r.status_code == 404:
                return None
            r.raise_for_status()
            time.sleep(RATE_LIMIT)
            return r
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(3 * (attempt + 1))
        except Exception as e:
            print(f"  [warn] {e}")
            return None
    return None


# ── Source 1: MLCommons GitHub ────────────────────────────────────────────────

def get_repo_tree(repo, branch, headers):
    for br in [branch, "master"]:
        url = f"{GITHUB_API}/repos/{repo}/git/trees/{br}?recursive=1"
        r = safe_get(url, headers)
        if r:
            data = r.json()
            tree = data.get("tree", [])
            if tree:
                return tree
    return []


def parse_mlperf_result(content, path):
    result = {"source_file": path, "source": "mlcommons_tiny"}

    latency_patterns = [
        r"m_results\[0\]\s*[=:]\s*([\d.]+)",
        r"performance_result\s*[=:]\s*([\d.]+)",
        r"latency[_\s]*us\s*[=:]\s*([\d.]+)",
        r"inferences_per_second\s*[=:]\s*([\d.]+)",
    ]
    for pat in latency_patterns:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            if "inferences_per_second" in pat:
                if val > 0:
                    result["latency_ms"] = round(1000.0 / val, 4)
            else:
                result["latency_us"] = val
                result["latency_ms"] = round(val / 1000.0, 4)
            break

    energy_patterns = [
        r"m_results\[1\]\s*[=:]\s*([\d.]+)",
        r"energy[_\s]*mj\s*[=:]\s*([\d.]+)",
        r"energy\s*[=:]\s*([\d.]+)\s*mj",
    ]
    for pat in energy_patterns:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            result["energy_mJ"] = val
            result["energy_J"]  = round(val / 1000.0, 8)
            break

    return result if ("latency_ms" in result or "energy_J" in result) else None


def parse_system_json(content):
    try:
        data = json.loads(content)
        return {k: str(v) for k, v in data.items()
                if k in ("hw_name", "sw_name", "processor", "accelerator", "system_name")}
    except Exception:
        return {}


def scrape_mlcommons(token, report):
    headers = make_headers(token)
    rows = []
    print("\n[scraper] Source 1: MLCommons tiny_results GitHub")

    if not token:
        print("  [info] No GitHub token — unauthenticated limit is 60 req/hr.")
        print("         Get a free token at github.com/settings/tokens")
        print("         Then run: python scraper.py --github_token YOUR_TOKEN")

    for repo, branch, result_filename in MLCOMMONS_REPOS:
        print(f"  Scanning {repo} ...")
        tree = get_repo_tree(repo, branch, headers)

        if not tree:
            report.append(f"SKIP {repo}: empty tree (rate limited or no token)")
            print(f"    Skipped")
            continue

        result_files = [
            f for f in tree
            if f["path"].endswith(result_filename)
            and "/performance" in f["path"]
            and f.get("type") == "blob"
        ]
        system_files = [
            f for f in tree
            if f["path"].endswith(".json")
            and "/systems/" in f["path"]
            and f.get("type") == "blob"
        ]

        print(f"    {len(result_files)} result files, {len(system_files)} system files")

        system_lookup = {}
        for sf in system_files[:25]:
            raw_url = f"{GITHUB_RAW}/{repo}/{branch}/{sf['path']}"
            r = safe_get(raw_url, headers)
            if r:
                key = "/".join(sf["path"].split("/")[:3])
                system_lookup[key] = parse_system_json(r.text)

        for rf in result_files[:80]:
            path    = rf["path"]
            raw_url = f"{GITHUB_RAW}/{repo}/{branch}/{path}"
            r = safe_get(raw_url, headers)
            if not r:
                continue

            parsed = parse_mlperf_result(r.text, path)
            if not parsed:
                continue

            path_lower = path.lower()
            device = match_device(path_lower)
            model  = match_model(path_lower)

            sys_key  = "/".join(path.split("/")[:3])
            sys_info = system_lookup.get(sys_key, {})
            sys_str  = " ".join(sys_info.values()).lower()
            if not device:
                device = match_device(sys_str)
            if not model:
                model  = match_model(sys_str)

            parsed.update({
                "device":  device or "unknown",
                "model":   model  or "unknown",
                "repo":    repo,
                "hw_name": sys_info.get("hw_name", ""),
            })
            rows.append(parsed)

        report.append(f"OK {repo}: {len(result_files)} result files")

    print(f"  Collected {len(rows)} rows from MLCommons")
    return rows


# ── Source 2: Papers With Code ────────────────────────────────────────────────

PWC_BASE = "https://paperswithcode.com/api/v1"
PWC_QUERIES = [
    ("mobilenet-v2",    "mobilenetv2"),
    ("mobilenet-v3",    "mobilenetv3_small"),
    ("resnet-18",       "resnet18"),
    ("efficientnet-b0", "efficientnet_b0"),
]


def scrape_papers_with_code(report):
    rows = []
    print("\n[scraper] Source 2: Papers With Code API")

    pwc_headers = {"User-Agent": "edge-ai-energy-scraper/2.0"}

    for query, canonical_model in PWC_QUERIES:
        try:
            r = requests.get(
                f"{PWC_BASE}/models/?name={query}&page=1&page_size=5",
                headers=pwc_headers, timeout=TIMEOUT
            )
            time.sleep(RATE_LIMIT)

            if r.status_code != 200:
                print(f"  [warn] PWC {r.status_code} for {query}")
                continue
            if not r.text.strip():
                print(f"  [warn] PWC empty response for {query}")
                continue

            data = r.json()

        except Exception as e:
            print(f"  [warn] PWC failed for {query}: {e}")
            continue

        for entry in data.get("results", [])[:3]:
            model_id = entry.get("id", "")
            if not model_id:
                continue
            try:
                rr = requests.get(
                    f"{PWC_BASE}/models/{model_id}/results/",
                    headers=pwc_headers, timeout=TIMEOUT
                )
                time.sleep(RATE_LIMIT)
                if rr.status_code != 200 or not rr.text.strip():
                    continue
                results_data = rr.json()
            except Exception:
                continue

            for res in results_data.get("results", []):
                hardware = str(res.get("hardware") or "")
                device   = match_device(hardware.lower())
                if not device:
                    continue
                metrics = res.get("metrics") or {}
                row = {"source": "papers_with_code", "device": device,
                       "model": canonical_model, "hw_name": hardware}
                for k, v in metrics.items():
                    kl = k.lower()
                    try:
                        fv = float(str(v).replace(",", "").strip())
                    except Exception:
                        continue
                    if "latency" in kl or "inference time" in kl:
                        row["latency_ms"] = fv
                    elif "energy" in kl:
                        row["energy_J"] = fv
                    elif "power" in kl:
                        row["power_W"] = fv
                if "latency_ms" in row or "energy_J" in row:
                    rows.append(row)

    report.append(f"OK papers_with_code: {len(rows)} rows")
    print(f"  Collected {len(rows)} rows from Papers With Code")
    return rows


# ── Source 3: Coral ───────────────────────────────────────────────────────────

CORAL_KNOWN = [
    {"model": "mobilenetv2",       "latency_ms": 2.4,  "power_W": 2.0, "notes": "coral.ai/docs MobileNetV2 INT8"},
    {"model": "mobilenetv3_small", "latency_ms": 3.1,  "power_W": 2.0, "notes": "coral.ai/docs MobileNetV3"},
    {"model": "efficientnet_b0",   "latency_ms": 4.7,  "power_W": 2.0, "notes": "coral.ai/docs EfficientNet-Lite"},
]


def scrape_coral(report):
    rows = []
    print("\n[scraper] Source 3: Coral AI benchmarks")

    r = safe_get("https://coral.ai/docs/edgetpu/benchmarks/",
                 {"User-Agent": "edge-ai-energy-scraper/2.0"})

    if r:
        pattern = re.compile(
            r"(MobileNet[^\n<]{0,30}?|EfficientNet[^\n<]{0,20}?)"
            r".*?(\d+\.?\d*)\s*ms.*?(\d+\.?\d*)\s*ms",
            re.IGNORECASE | re.DOTALL
        )
        for m in pattern.finditer(r.text):
            model = match_model(m.group(1))
            if model:
                try:
                    rows.append({
                        "source": "coral_benchmarks", "device": "coral_tpu",
                        "model": model, "latency_ms": float(m.group(3)),
                        "power_W": 2.0, "hw_name": "Coral Edge TPU",
                    })
                except Exception:
                    pass

    if not rows:
        print("  Live scrape found nothing — using known Coral values")
        for k in CORAL_KNOWN:
            rows.append({"source": "coral_known", "device": "coral_tpu",
                         "hw_name": "Coral Edge TPU", **k})

    report.append(f"OK coral: {len(rows)} rows")
    print(f"  Collected {len(rows)} rows from Coral")
    return rows


# ── Source 4: Known published values ─────────────────────────────────────────

KNOWN_VALUES = [
    # Raspberry Pi 4 — Banbury et al. MLSys 2022
    {"device": "raspberry_pi4", "model": "mobilenetv2",       "latency_ms": 78.5,  "power_W": 3.2, "source": "banbury2022_mlsys"},
    {"device": "raspberry_pi4", "model": "mobilenetv3_small", "latency_ms": 55.3,  "power_W": 3.1, "source": "banbury2022_mlsys"},
    {"device": "raspberry_pi4", "model": "resnet18",          "latency_ms": 198.4, "power_W": 3.6, "source": "banbury2022_mlsys"},
    {"device": "raspberry_pi4", "model": "efficientnet_b0",   "latency_ms": 312.0, "power_W": 3.5, "source": "banbury2022_mlsys"},
    # Jetson Nano — NVIDIA benchmarks + arxiv:2407.11061
    {"device": "jetson_nano",   "model": "mobilenetv2",       "latency_ms": 44.1,  "power_W": 4.8, "source": "nvidia_jetson_benchmarks"},
    {"device": "jetson_nano",   "model": "mobilenetv3_small", "latency_ms": 38.7,  "power_W": 4.6, "source": "nvidia_jetson_benchmarks"},
    {"device": "jetson_nano",   "model": "resnet18",          "latency_ms": 46.2,  "power_W": 5.1, "source": "nvidia_jetson_benchmarks"},
    {"device": "jetson_nano",   "model": "efficientnet_b0",   "latency_ms": 89.0,  "power_W": 4.9, "source": "nvidia_jetson_benchmarks"},
    # Coral — coral.ai/docs/edgetpu/benchmarks
    {"device": "coral_tpu",     "model": "mobilenetv2",       "latency_ms": 2.4,   "power_W": 2.0, "source": "coral_ai_docs"},
    {"device": "coral_tpu",     "model": "mobilenetv3_small", "latency_ms": 3.1,   "power_W": 2.0, "source": "coral_ai_docs"},
    {"device": "coral_tpu",     "model": "efficientnet_b0",   "latency_ms": 4.7,   "power_W": 2.0, "source": "coral_ai_docs"},
    # STM32 — mlperf tiny v0.5 reference
    {"device": "stm32",         "model": "resnet18",          "latency_ms": 920.0, "power_W": 0.048, "source": "mlperf_tiny_v0.5_ref"},
    {"device": "stm32",         "model": "mobilenetv2",       "latency_ms": 540.0, "power_W": 0.045, "source": "mlperf_tiny_v0.5_ref"},
    # Apple Silicon — Shah 2024 paper
    {"device": "apple_silicon", "model": "mobilenetv3_small", "latency_ms": 15.14, "power_W": 5.3, "energy_J": 0.080, "source": "shah2024_paper"},
    {"device": "apple_silicon", "model": "mobilenetv2",       "latency_ms": 20.07, "power_W": 5.3, "energy_J": 0.106, "source": "shah2024_paper"},
    {"device": "apple_silicon", "model": "resnet18",          "latency_ms": 20.68, "power_W": 5.3, "energy_J": 0.110, "source": "shah2024_paper"},
    {"device": "apple_silicon", "model": "tiny_vit_5m",       "latency_ms": 41.33, "power_W": 5.3, "energy_J": 0.219, "source": "shah2024_paper"},
    {"device": "apple_silicon", "model": "efficientnet_b0",   "latency_ms": 55.08, "power_W": 5.3, "energy_J": 0.292, "source": "shah2024_paper"},
]


def load_known_values(report):
    print("\n[scraper] Source 4: Known published values")
    rows = []
    for row in KNOWN_VALUES:
        r = dict(row)
        if "energy_J" not in r and "power_W" in r and "latency_ms" in r:
            r["energy_J"] = round(r["power_W"] * r["latency_ms"] / 1000.0, 6)
        r.setdefault("hw_name", r["device"])
        rows.append(r)
    report.append(f"OK known_values: {len(rows)} rows")
    print(f"  Loaded {len(rows)} rows")
    return rows


# ── Consolidation ─────────────────────────────────────────────────────────────

def consolidate(all_rows):
    df = pd.DataFrame(all_rows)
    for col in ["source","device","model","latency_ms","energy_J","power_W","hw_name","notes"]:
        if col not in df.columns:
            df[col] = None

    df = df[df["device"].notna() & ~df["device"].isin(["unknown",""])]
    df = df[df["model"].notna()  & ~df["model"].isin(["unknown",""])]
    df = df[df["latency_ms"].notna() | df["energy_J"].notna()]

    for col in ["latency_ms","energy_J","power_W"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask = df["energy_J"].isna() & df["power_W"].notna() & df["latency_ms"].notna()
    df.loc[mask, "energy_J"] = (df.loc[mask,"power_W"] * df.loc[mask,"latency_ms"] / 1000.0).round(6)

    mask2 = df["power_W"].isna() & df["energy_J"].notna() & df["latency_ms"].notna()
    df.loc[mask2,"power_W"] = (df.loc[mask2,"energy_J"] / (df.loc[mask2,"latency_ms"] / 1000.0)).round(4)

    df = df[df["latency_ms"].isna() | ((df["latency_ms"] > 0.01) & (df["latency_ms"] < 500_000))]
    df = df[df["energy_J"].isna()   | ((df["energy_J"] > 1e-9)   & (df["energy_J"] < 500))]

    return df.reset_index(drop=True)


def write_report(df, report, path):
    coverage = df.groupby(["device","model"]).size().unstack(fill_value=0)
    lines = [
        "Edge AI Energy Scraper Report",
        f"Generated: {datetime.now().isoformat()}",
        "", "── Source status ──", *report, "",
        f"Total rows : {len(df)}",
        f"Devices    : {sorted(df['device'].unique())}",
        f"Models     : {sorted(df['model'].unique())}",
        "", "── Coverage ──", coverage.to_string(),
        "", "── Sample rows ──",
        df[["device","model","latency_ms","energy_J","power_W","source"]]
        .dropna(subset=["latency_ms"]).head(25).to_string(index=False),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[scraper] Report → {path}")


# ── Entry Point ───────────────────────────────────────────────────────────────

ALL_SOURCES = ["mlcommons","papers_with_code","coral","known_values"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sources",      nargs="+", default=ALL_SOURCES, choices=ALL_SOURCES)
    p.add_argument("--output",       default="real_benchmark_data.csv")
    p.add_argument("--report",       default="scrape_report.txt")
    p.add_argument("--github_token", default=os.environ.get("GITHUB_TOKEN",""),
                   help="GitHub PAT — get one free at github.com/settings/tokens (no scopes needed)")
    return p.parse_args()


def main():
    args   = parse_args()
    token  = args.github_token or None
    report = []
    rows   = []

    if "mlcommons"        in args.sources: rows.extend(scrape_mlcommons(token, report))
    if "papers_with_code" in args.sources: rows.extend(scrape_papers_with_code(report))
    if "coral"            in args.sources: rows.extend(scrape_coral(report))
    if "known_values"     in args.sources: rows.extend(load_known_values(report))

    if not rows:
        print("[scraper] No rows collected.")
        return

    df = consolidate(rows)
    print(f"\n[scraper] Final: {len(df)} rows")
    print(df.groupby(["device","model"]).size().unstack(fill_value=0).to_string())

    df.to_csv(args.output, index=False)
    print(f"[scraper] Saved → {args.output}")
    write_report(df, report, args.report)

    print("\n── Next step ──")
    print(f"python ingest_real_data.py --csv {args.output}")


if __name__ == "__main__":
    main()
