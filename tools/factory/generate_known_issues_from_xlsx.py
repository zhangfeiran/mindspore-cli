#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import os
import re
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ROOT = Path(__file__).resolve().parents[2]
KNOWN_ISSUES_DIR = ROOT / "incubating" / "factory" / "cards" / "known_issues" / "generated"
PACK_PATH = ROOT / "incubating" / "factory" / "manifests" / "pack.yaml"
XLSX_PATH = ROOT.parent / "gitcode问题单26.1-26.4.xlsx"

MANUAL_KNOWN_ISSUES = {
    "function-operator-dsa-not-implemented": "known_issues/function-operator-dsa-not-implemented.yaml",
    "compat-operator-flash-attention-version-mismatch": "known_issues/compat-operator-flash-attention-version-mismatch.yaml",
    "accuracy-operator-softmax-fp16-drift": "known_issues/accuracy-operator-softmax-fp16-drift.yaml",
    "accuracy-codetrace-grad-reducer-inline": "known_issues/accuracy-codetrace-grad-reducer-inline.yaml",
}

SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

ISSUE_PRIORITY_MAP = {
    "无优先级": "low",
    "次要": "medium",
    "主要": "high",
    "严重": "critical",
}


def excel_col_to_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch.upper()) - ord("A") + 1)
    return n - 1


def read_sheet_rows(path: Path) -> List[Dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        shared = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", NS):
                shared.append("".join(t.text or "" for t in si.iterfind(".//a:t", NS)))

        wb = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        relmap = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        first_sheet = wb.find("a:sheets", NS).findall("a:sheet", NS)[0]
        rid = first_sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        ws = ET.fromstring(zf.read("xl/" + relmap[rid]))
        rows = ws.findall(".//a:sheetData/a:row", NS)

        parsed: List[Dict[int, str]] = []
        for row in rows:
            cols: Dict[int, str] = {}
            for c in row.findall("a:c", NS):
                idx = excel_col_to_index(c.attrib["r"])
                t = c.attrib.get("t")
                v = c.find("a:v", NS)
                val = ""
                if v is not None:
                    val = v.text or ""
                    if t == "s":
                        val = shared[int(val)]
                cols[idx] = val
            parsed.append(cols)

        headers = [parsed[0].get(i, "").strip() for i in range(max(parsed[0].keys()) + 1)]
        data: List[Dict[str, str]] = []
        for row in parsed[1:]:
            entry = {headers[i]: row.get(i, "").strip() for i in range(len(headers)) if headers[i]}
            if any(entry.values()):
                data.append(entry)
        return data


def normalize_platforms(raw: str) -> List[str]:
    parts = re.split(r"[,/，、\s]+", raw or "")
    result = []
    seen = set()
    for p in parts:
        if not p:
            continue
        pl = p.lower()
        pl = {
            "ascend": "ascend",
            "cpu": "cpu",
            "gpu": "gpu",
        }.get(pl, pl)
        if pl not in seen:
            seen.add(pl)
            result.append(pl)
    return result or ["unknown"]


def map_symptom(title: str) -> str:
    t = title.lower()
    if any(k in title for k in ["精度", "异常值", "不一致", "未对齐", "误差", "对不上", "不符", "差太大"]) or any(k in t for k in ["nan", "inf", "loss", "acc", "accuracy", "mismatch"]):
        return "accuracy"
    if any(k in title for k in ["性能", "耗时", "吞吐", "内存"]) or any(k in t for k in ["tps", "throughput", "latency", "performance", "generate speed", "compile_time", "fps"]):
        return "performance"
    return "failure"


def slugify(text: str) -> str:
    text = text.lower()
    text = text.replace("mindformers", "mindformers")
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "generic"


def extract_operator(title: str) -> str | None:
    lower = title.lower()
    for pattern in [
        r"(?<![a-z0-9_])ops\.([a-z0-9_]+)(?![a-z0-9_])",
        r"(?<![a-z0-9_])nn\.([a-z0-9_]+)(?![a-z0-9_])",
        r"(?<![a-z0-9_])mint\.([a-z0-9_]+)(?![a-z0-9_])",
        r"(?<![a-z0-9_])tensor\.([a-z0-9_]+)(?![a-z0-9_])",
        r"(?<![a-z0-9_])([a-z0-9_]+loss)(?![a-z0-9_])",
        r"(?<![a-z0-9_])(reversesequence|nanmean|allfinite|softmax|atan2|atan|sqrt|equal|searchsorted|upsamplenearest2dbackward|upsampletrilinear3d|flash_attention_score|flashattentionscore|gru|asgd|kldivloss|ctcloss|focalloss|baddbmm|leftshift|reciprocal|mul|mod|cumsum|sumext|logical_and|histogram|jacfwd|frombuffer|ringattentionupdate|batchinvariant|dsa|cos)(?![a-z0-9_])",
    ]:
        m = re.search(pattern, lower)
        if m:
            token = m.group(1)
            return token.replace("_", "-")
    # preserve CamelCase operator names from titles like KLDivLoss / CTCLoss / Conv1d
    for pattern in [
        r"(?<![A-Za-z0-9_])(KLDivLoss|CTCLoss|CtcLoss|Conv1d|GRU|ASGD|DSA|MoELayer|SharedExpertMLP|ReverseSequence|SumExt|RingAttentionUpdate)(?![A-Za-z0-9_])",
        r"(?<![A-Za-z0-9_])(flash_attention_score)(?![A-Za-z0-9_])",
    ]:
        m = re.search(pattern, title)
        if m:
            return slugify(m.group(1))
    return None


def classify_issue(title: str, component: str, project: str) -> Tuple[str, str, List[str], List[str], str]:
    lower = title.lower()
    base_tags = [slugify(component) if component else "unknown-component", slugify(project) if project else "unknown-project"]

    if title.startswith("CVE-"):
        return (
            "function-security-cve-tracking",
            "failure",
            base_tags + ["security", "cve"],
            [],
            r"CVE-\d{4}-\d+",
        )

    if "aclnn" in lower and ("workspace" in lower or "call failed" in lower):
        op = extract_operator(title)
        if not op:
            m = re.search(r"aclnn([a-z0-9]+)", lower)
            op = slugify(m.group(1)) if m else "runtime"
        return (
            f"compat-operator-{op}-workspace-failure",
            "failure",
            base_tags + ["aclnn", "workspace", op],
            [op] if op != "runtime" else [],
            r"aclnn[A-Za-z0-9]+(GetWorkspaceSize| call failed)|WorkspaceSize call failed",
        )

    if (
        "segmentation fault" in lower
        or "double free" in lower
        or re.search(r"(引起的core|偶现core|执行core|发生core|发生Segmentation fault|core dump|dumped core)", title, re.IGNORECASE)
    ):
        scope = "meta" if "meta" in lower else "runtime"
        return (
            f"failure-{scope}-segfault-crash",
            "failure",
            base_tags + [scope, "segfault", "crash"],
            [],
            r"(Segmentation fault|double free|core)",
        )

    if "timeout" in lower or "超时" in title:
        return (
            "failure-runtime-timeout",
            "failure",
            base_tags + ["runtime", "timeout"],
            [],
            r"(timeout|超时)",
        )

    if "warning日志" in title or "warning log" in lower or "频繁打印warning" in title:
        return (
            "function-runtime-warning-log-noise",
            "failure",
            base_tags + ["runtime", "warning-log"],
            [],
            r"(warning日志|warning log|频繁打印warning)",
        )

    if "告警日志过多" in title or "warning日志重复次数" in title:
        return (
            "function-runtime-warning-log-noise",
            "failure",
            base_tags + ["runtime", "warning-log"],
            [],
            r"(告警日志过多|WARNING日志重复次数)",
        )

    if "日志或者注释中存在" in title or "中文版格式有问题" in title or "示例说明" in title:
        return (
            "function-doc-runtime-docstring-mismatch",
            "failure",
            base_tags + ["doc", "runtime-doc"],
            [],
            r"(日志或者注释中存在|中文版格式有问题|示例说明)",
        )

    if "编译时间" in title and "校验" in title:
        return (
            "failure-compile-time-check-flaky",
            "failure",
            base_tags + ["compile-time", "flaky"],
            [],
            r"(编译时间|compile_time)",
        )

    if "*args" in title or "**kwargs" in title:
        return (
            "function-compiler-varargs-unsupported",
            "failure",
            base_tags + ["compiler", "varargs"],
            [],
            r"(\\*args|\\*\\*kwargs)",
        )

    if "compress-optimizer" in lower:
        return (
            "function-training-compress-optimizer-config-invalid",
            "failure",
            base_tags + ["training", "compress-optimizer", "config"],
            [],
            r"compress-optimizer",
        )

    if "图算融合" in title or ("融合失败" in title and "o1" in lower):
        return (
            "failure-compile-fusion-path-missing",
            "failure",
            base_tags + ["compile", "fusion"],
            [],
            r"(图算融合|融合失败|O1)",
        )

    if "memory not enough" in lower or "out of memory" in lower:
        return (
            "failure-runtime-memory-oom",
            "failure",
            base_tags + ["runtime", "memory", "oom"],
            [],
            r"(Memory not enough|out of memory)",
        )

    if "memory_tracker" in lower:
        return (
            "function-runtime-memory-tracker-config-invalid",
            "failure",
            base_tags + ["runtime", "memory-tracker", "config"],
            [],
            r"memory_tracker",
        )

    if "tsan" in lower or "threadsanitizer" in lower or "data race" in lower:
        return (
            "failure-runtime-thread-sanitizer-race",
            "failure",
            base_tags + ["runtime", "tsan", "race"],
            [],
            r"(TSAN|ThreadSanitizer|Data race)",
        )

    if "dsa" in lower and any(k in title for k in ["示例说明", "已知问题"]):
        return (
            "function-operator-dsa-not-implemented",
            "failure",
            base_tags + ["dsa", "unsupported"],
            ["dsa"],
            r"(DSA已知问题|DSA.*示例说明)",
        )

    if "pylint" in lower:
        return (
            "function-test-pylint-legacy-regression",
            "failure",
            base_tags + ["test", "pylint", "legacy"],
            [],
            r"pylint",
        )

    if "bind_numa" in lower or "绑核" in title or "localrank" in lower or "局部rank" in title:
        return (
            "function-runtime-msrun-config-invalid",
            "failure",
            base_tags + ["msrun", "config"],
            [],
            r"(bind_numa|绑核|localrank|局部rank|msrun)",
        )

    if "算力切分" in title:
        return (
            "failure-parallel-distributed-runtime",
            "failure",
            base_tags + ["parallel", "distributed"],
            [],
            r"(算力切分|parallel|distributed)",
        )

    if "link" in lower and "跳转错误" in title:
        return (
            "function-doc-link-broken",
            "failure",
            base_tags + ["doc", "link"],
            [],
            r"(跳转错误|link)",
        )

    if "导出mindir" in title or "导出midir" in lower or "mindir模型" in title:
        return (
            "function-runtime-mindir-export-load-failure",
            "failure",
            base_tags + ["mindir", "export-load"],
            [],
            r"(mindir|导出MINDIR|导出mindir)",
        )

    if "import问题" in title:
        return (
            "function-runtime-module-import-missing",
            "failure",
            base_tags + ["runtime", "import", "module-missing"],
            [],
            r"(import问题|ModuleNotFoundError|No module named|ImportError)",
        )

    if "onnx" in lower or "exportprimitive" in lower or "convert map" in lower:
        return (
            "function-runtime-export-convert-map-missing",
            "failure",
            base_tags + ["export", "convert-map"],
            [],
            r"(onnx|ExportPrimitive|convert map)",
        )

    if "triton" in lower:
        return (
            "compat-runtime-triton-backend-integration",
            "failure",
            base_tags + ["triton", "backend"],
            [],
            r"triton",
        )

    if "hccl" in lower and "group name" in lower:
        return (
            "failure-parallel-hccl-group-config-invalid",
            "failure",
            base_tags + ["parallel", "hccl", "group"],
            [],
            r"(hccl|group name)",
        )

    if "communicator of group" in lower:
        return (
            "failure-parallel-hccl-group-config-invalid",
            "failure",
            base_tags + ["parallel", "hccl", "group"],
            [],
            r"(Communicator of group|group.*inited: failed)",
        )

    if "recompute.cc" in lower or "recompute" in lower and "warning" in lower:
        return (
            "function-runtime-recompute-warning-noise",
            "failure",
            base_tags + ["recompute", "warning-log"],
            [],
            r"(recompute\.cc|recompute)",
        )

    if "parse dynamic kernel config fail" in lower:
        op = extract_operator(title) or "runtime"
        return (
            f"compat-operator-{op}-dynamic-kernel-config-fail",
            "failure",
            base_tags + ["dynamic-kernel-config", op],
            [op] if op != "runtime" else [],
            r"Parse dynamic kernel config fail",
        )

    if "call aclnn" in lower and "failed" in lower:
        op = extract_operator(title) or "runtime"
        return (
            f"failure-operator-{op}-kernel-launch-failure" if op != "runtime" else "failure-runtime-kernel-launch-failure",
            "failure",
            base_tags + ["kernel-launch", op],
            [op] if op != "runtime" else [],
            r"Call aclnn[A-Za-z0-9]+ failed",
        )

    if "cannot get adapter for" in lower:
        return (
            "function-runtime-adapter-missing",
            "failure",
            base_tags + ["adapter", "runtime"],
            [],
            r"Cannot get adapter for",
        )

    if "aicpu kernel bin" in lower:
        op = extract_operator(title) or "runtime"
        return (
            f"failure-operator-{op}-aicpu-kernel-missing" if op != "runtime" else "failure-runtime-aicpu-kernel-missing",
            "failure",
            base_tags + ["aicpu-kernel", op],
            [op] if op != "runtime" else [],
            r"(aicpu kernel bin|cust_aicpu_kernel)",
        )

    if "state_dict" in lower or "load_checkpoint" in lower or "加载权重" in title or "ckpt" in lower or "safetensors" in lower:
        return (
            "function-runtime-checkpoint-load-failure",
            "failure",
            base_tags + ["checkpoint", "load"],
            [],
            r"(state_dict|load_checkpoint|加载权重|ckpt|safetensors)",
        )

    if "devicectx('meta')" in lower or "device=meta" in lower or ("meta" in lower and any(k in lower for k in ["pointer", "nonzero", "rand", "empty", "conv", "lazy async copy", "backend type is invalid"])):
        return (
            "function-runtime-meta-device-unsupported",
            "failure",
            base_tags + ["meta", "device"],
            [],
            r"(meta|DeviceCtx\\('meta'\\)|device=meta)",
        )

    if "unsupported" in lower and "complex64" in lower:
        op = extract_operator(title) or "runtime"
        return (
            f"compat-operator-{op}-complex64-unsupported",
            "failure",
            base_tags + ["complex64", "unsupported", op],
            [op] if op != "runtime" else [],
            r"(unsupported|不支持).*complex64",
        )

    if "does not match expected type" in lower or "输入类型没对齐" in title or "输入类型不匹配" in title or "different shape" in lower:
        op = extract_operator(title) or "runtime"
        return (
            f"function-operator-{op}-type-shape-mismatch" if op != "runtime" else "function-runtime-type-shape-mismatch",
            "failure",
            base_tags + ["type-shape-mismatch", op],
            [op] if op != "runtime" else [],
            r"(type does not match expected type|输入类型没对齐|different shape)",
        )

    if any(k in lower for k in ["launch kernel failed", "op execute failed", "aicore kernel execute failed", "inittilingparsectx failed", "acl compile and execute failed"]):
        op = extract_operator(title) or "runtime"
        return (
            f"failure-operator-{op}-kernel-launch-failure" if op != "runtime" else "failure-runtime-kernel-launch-failure",
            "failure",
            base_tags + ["kernel-launch", op],
            [op] if op != "runtime" else [],
            r"(Launch kernel failed|Op execute failed|Aicore kernel execute failed|InitTilingParseCtx failed|Acl compile and execute failed)",
        )

    if "执行卡住" in title or "卡住" in title or "hang" in lower:
        return (
            "failure-runtime-hang",
            "failure",
            base_tags + ["runtime", "hang"],
            [],
            r"(执行卡住|卡住|hang)",
        )

    if "资料" in title or "说明文档" in title or "文档链接" in title or "跳转错误" in title:
        return (
            "function-doc-documentation-missing",
            "failure",
            base_tags + ["doc", "documentation"],
            [],
            r"(资料|说明文档|文档链接|跳转错误)",
        )

    if "cc1plus:" in lower or "error: expected declaration before" in lower or "cmake" in lower and "无法编译" in title:
        return (
            "failure-build-toolchain-compile-error",
            "failure",
            base_tags + ["build", "toolchain"],
            [],
            r"(cc1plus:|expected declaration before|cmake)",
        )

    if "ascend_home_path not found" in lower:
        return (
            "function-runtime-env-var-missing",
            "failure",
            base_tags + ["env", "var-missing"],
            [],
            r"ASCEND_HOME_PATH not found",
        )

    if "config_path" in lower and "msprobe" in lower:
        return (
            "function-runtime-config-path-missing",
            "failure",
            base_tags + ["config-path", "msprobe"],
            [],
            r"config_path",
        )

    if "device id is valid" in lower or "aclrtsetdevice failed" in lower:
        return (
            "function-runtime-device-config-invalid",
            "failure",
            base_tags + ["device-config", "runtime"],
            [],
            r"(aclrtSetDevice failed|device id is valid)",
        )

    if "index is out of bounds" in lower or "indexerror" in lower:
        return (
            "function-runtime-index-out-of-bounds",
            "failure",
            base_tags + ["runtime", "index"],
            [],
            r"(IndexError|out of bounds)",
        )

    if "zbv" in lower:
        return (
            "function-training-zbv-config-invalid",
            "failure",
            base_tags + ["training", "zbv", "config"],
            [],
            r"zbv",
        )

    if "tnd模式" in title:
        return (
            "function-training-dsa-config-constraint",
            "failure",
            base_tags + ["training", "dsa", "config"],
            ["dsa"],
            r"(tnd模式|DSA特性)",
        )

    if "测试用例未适配" in title or "ut修复" in title or "l1用例报错" in title or "legacy用例报错" in title or "用例报错" in title:
        return (
            "failure-test-adaptation-missing",
            "failure",
            base_tags + ["test", "adaptation"],
            [],
            r"(测试用例未适配|ut修复|L1用例报错|legacy用例报错)",
        )

    if "升级依赖包版本" in title:
        return (
            "compat-runtime-dependency-version-mismatch",
            "failure",
            base_tags + ["runtime", "dependency", "version"],
            [],
            r"升级依赖包版本",
        )

    if "高可用" in title and "信号未上报" in title:
        return (
            "failure-highavailability-signal-report-missing",
            "failure",
            base_tags + ["highavailability", "signal"],
            [],
            r"(高可用|信号未上报)",
        )

    if "高可用" in title and "loss后报错" in title:
        return (
            "failure-highavailability-recovery-runtime",
            "failure",
            base_tags + ["highavailability", "recovery", "runtime"],
            [],
            r"(高可用|loss后报错)",
        )

    if "acl stream synchronize failed" in lower:
        return (
            "failure-runtime-stream-synchronize-failed",
            "failure",
            base_tags + ["runtime", "stream-sync"],
            [],
            r"ACL stream synchronize failed",
        )

    if "确定性固定不住" in title:
        return (
            "accuracy-training-determinism-drift",
            "accuracy",
            base_tags + ["accuracy", "determinism", "training"],
            [],
            r"(确定性固定不住|determin)",
        )

    if "loss有多个突刺" in title:
        return (
            "accuracy-training-loss-spike-drift",
            "accuracy",
            base_tags + ["accuracy", "training", "loss-spike"],
            [],
            r"(loss有多个突刺|loss spike)",
        )

    if "loss误差超" in title or "精度和现场精度不一致" in title or "精度异常" in title and any(k in lower for k in ["qwen3vl", "telecha", "deepseek", "mcore"]):
        return (
            "accuracy-training-model-loss-drift",
            "accuracy",
            base_tags + ["accuracy", "training", "model-loss"],
            [],
            r"(loss误差超|精度和现场精度不一致|精度异常)",
        )

    if "cann升级" in lower and "精度错误" in title:
        return (
            "compat-runtime-cann-upgrade-precision-drift",
            "accuracy",
            base_tags + ["accuracy", "cann-upgrade", "compat"],
            [],
            r"(CANN升级|精度错误)",
        )

    if "更改其中一个的激活函数" in title:
        return (
            "accuracy-runtime-activation-update-state-drift",
            "accuracy",
            base_tags + ["accuracy", "activation", "state-update"],
            [],
            r"(激活函数|精度问题)",
        )

    if "dpmsolversdescheduler" in lower:
        return (
            "accuracy-model-diffusers-scheduler-regression",
            "accuracy",
            base_tags + ["accuracy", "model", "diffusers", "scheduler"],
            [],
            r"DPMSolverSDEScheduler",
        )

    if "缺少精度用例" in title:
        return (
            "failure-test-accuracy-case-missing",
            "failure",
            base_tags + ["test", "accuracy-case"],
            [],
            r"缺少精度用例",
        )

    if "eval_indexes" in lower or "metrics" in lower and "参数不一致" in title:
        return (
            "function-runtime-metric-input-mismatch",
            "failure",
            base_tags + ["runtime", "metric", "input-mismatch"],
            [],
            r"(eval_indexes|metrics)",
        )

    if "返回参数" in title and "接收参数不一致" in title:
        return (
            "function-runtime-interface-contract-mismatch",
            "failure",
            base_tags + ["runtime", "interface-contract"],
            [],
            r"(返回参数|接收参数不一致)",
        )

    if "报错信息优化" in title or "跟预期不一致" in title and "参数名" in title:
        return (
            "function-runtime-error-message-mismatch",
            "failure",
            base_tags + ["runtime", "error-message"],
            [],
            r"(报错信息优化|未知参数名|跟预期不一致)",
        )

    if "没有报错" in title or "报错不准确" in title:
        return (
            "function-runtime-error-message-mismatch",
            "failure",
            base_tags + ["runtime", "error-message"],
            [],
            r"(没有报错|报错不准确)",
        )

    if "batchinvariant" in lower and ("没有安装" in title or "明确的提示" in title):
        return (
            "compat-operator-batchinvariant-install-missing",
            "failure",
            base_tags + ["batchinvariant", "install"],
            ["batchinvariant"],
            r"(batchinvariant|没有安装)",
        )

    if "模型转换/推理失败" in title or "服务化推理" in title or "推理失败" in title:
        return (
            "function-model-inference-runtime-failure",
            "failure",
            base_tags + ["model", "inference", "runtime"],
            [],
            r"(模型转换/推理失败|服务化推理|推理失败)",
        )

    if "not support" in lower or "不支持" in title:
        op = extract_operator(title) or "runtime"
        return (
            f"compat-operator-{op}-unsupported" if op != "runtime" else "compat-runtime-unsupported",
            "failure",
            base_tags + ["unsupported", op],
            [op] if op != "runtime" else [],
            r"(not support|不支持|not implemented)",
        )

    if "unsupported op [allfinite]" in lower:
        return (
            "accuracy-codetrace-grad-reducer-inline",
            "failure",
            base_tags + ["codetrace", "allfinite", "grad-reducer"],
            ["allfinite", "grad-reducer"],
            r"Unsupported op \[AllFinite\] on CPU|Code trace accuracy is not 1\.0|self\.grad_reducer",
        )

    if "flashattentionscore" in lower and "not found" in lower:
        return (
            "compat-operator-flash-attention-version-mismatch",
            "failure",
            base_tags + ["flash-attention", "cann", "compat"],
            ["flash-attention"],
            r"FlashAttentionScore.*not found|kernel not found",
        )

    if "softmax" in lower and "fp16" in lower and any(k in lower for k in ["drift", "accuracy", "regression"]):
        return (
            "accuracy-operator-softmax-fp16-drift",
            "accuracy",
            base_tags + ["softmax", "fp16", "drift"],
            ["softmax"],
            r"softmax.*(fp16|float16).*(drift|accuracy|regression)",
        )

    if "全0" in title and "精度错误" in title:
        op = extract_operator(title) or "runtime"
        return (
            f"accuracy-operator-{op}-all-zero-output" if op != "runtime" else "accuracy-runtime-all-zero-output",
            "accuracy",
            base_tags + ["accuracy", "all-zero", op],
            [op] if op != "runtime" else [],
            r"(精度错误|全0)",
        )

    if "全为0" in title:
        op = extract_operator(title) or "runtime"
        return (
            f"accuracy-operator-{op}-all-zero-output" if op != "runtime" else "accuracy-runtime-all-zero-output",
            "accuracy",
            base_tags + ["accuracy", "all-zero", op],
            [op] if op != "runtime" else [],
            r"(全为0|全0)",
        )

    if "reshape入参为key" in title:
        return (
            "function-operator-flash-attention-arg-mapping-invalid",
            "failure",
            base_tags + ["flash-attention", "arg-mapping"],
            ["flash-attention"],
            r"(flash attention|reshape入参为key)",
        )

    if "回黄每日构建" in title and "mindformers" in lower and "用例失败" in title:
        return (
            "failure-test-mindformers-daily-case-failure",
            "failure",
            base_tags + ["mindformers", "daily", "testcase"],
            [],
            r"回黄每日构建.*MindFormers.*用例失败",
        )

    if "门禁" in title and ("用例失败" in title or "用例报错" in title or "test_" in lower):
        return (
            "failure-test-gate-case-failure",
            "failure",
            base_tags + ["gate", "testcase"],
            [],
            r"门禁.*(用例失败|用例报错|test_)",
        )

    if "profiler" in lower and "framework" in lower and any(k in title for k in ["为空", "缺少"]):
        return (
            "failure-profiler-framework-data-missing",
            "failure",
            base_tags + ["profiler", "framework", "data-missing"],
            [],
            r"profiler.*FRAMEWORK.*(为空|缺少)",
        )

    if "profiler" in lower:
        return (
            "failure-profiler-testcase-failure",
            "failure",
            base_tags + ["profiler", "testcase"],
            [],
            r"profiler.*(失败|报错|empty|missing)",
        )

    if any(k in lower for k in ["compile graph", "initialize ge failed", "kernel_graph"]) or any(k in title for k in ["编译失败", "编译耗时", "GE报错", "模拟编译"]):
        return (
            "failure-compile-graph-compile-failure",
            "failure",
            base_tags + ["compile", "graph"],
            [],
            r"(Compile graph|Initialize GE failed|编译失败|GE报错|模拟编译)",
        )

    if any(k in title for k in ["性能", "耗时", "吞吐", "内存", "低于基线"]) or any(k in lower for k in ["tps", "throughput", "latency", "performance", "劣化", "generate speed", "compile_time"]):
        model = "runtime"
        for cand in ["qwen3", "llama3", "deepseek", "mixtral", "transformer", "pangu", "flash_attention", "flashattention"]:
            if cand in lower:
                model = cand.replace("_", "-")
                break
        return (
            f"performance-training-{model}-regression",
            "performance",
            base_tags + ["performance", model],
            [],
            r"(性能|耗时|吞吐|内存|tps|throughput|latency|劣化)",
        )

    if any(k in lower for k in ["parallel", "alltoall", "pipeline", "world_size", "hyper_parallel", "load_parallel_checkpoint", "semi_auto_parallel", "dualpipv", "dualpipev"]) or any(k in title for k in ["多卡", "通信重排", "优化器并行", "并行"]):
        return (
            "failure-parallel-distributed-runtime",
            "failure",
            base_tags + ["parallel", "distributed"],
            [],
            r"(parallel|alltoall|pipeline|world_size|hyper_parallel|load_parallel_checkpoint|semi_auto_parallel|多卡|并行)",
        )

    if any(k in lower for k in ["modulenotfounderror", "no module named", "importerror"]):
        return (
            "function-runtime-module-import-missing",
            "failure",
            base_tags + ["runtime", "import", "module-missing"],
            [],
            r"(ModuleNotFoundError|No module named|ImportError)",
        )

    if any(k in lower for k in ["attributeerror", "has no attribute"]):
        return (
            "function-runtime-attribute-missing",
            "failure",
            base_tags + ["runtime", "attribute"],
            [],
            r"(AttributeError|has no attribute)",
        )

    if "not callable" in lower or "typeerror" in lower:
        return (
            "function-runtime-not-callable",
            "failure",
            base_tags + ["runtime", "typeerror"],
            [],
            r"(not callable|TypeError)",
        )

    operator = extract_operator(title)
    symptom = map_symptom(title)
    if symptom == "accuracy":
        if "控制流场景" in title and "融合" in title:
            return (
                "accuracy-compile-fusion-controlflow-drift",
                "accuracy",
                base_tags + ["accuracy", "compile", "fusion", "controlflow"],
                [],
                r"(控制流场景|融合)",
            )
        if "load_balancing_loss" in lower:
            return (
                "accuracy-training-load-balancing-loss-drift",
                "accuracy",
                base_tags + ["accuracy", "load-balancing-loss", "training"],
                [],
                r"load_balancing_loss|load balancing loss",
            )
        if "megatron" in lower and ("不一致" in title or "对齐" in title):
            op = operator or "runtime"
            return (
                f"accuracy-{('operator-' + op) if operator else 'training-runtime'}-megatron-mismatch",
                "accuracy",
                base_tags + ["accuracy", "megatron", op],
                [op] if operator else [],
                r"(megatron|对齐|不一致)",
            )
        if "determin" in lower or "确定性" in title:
            return (
                "accuracy-training-determinism-drift",
                "accuracy",
                base_tags + ["accuracy", "determinism", "training"],
                [],
                r"(确定性|determin)",
            )
        if "amp.autocast" in lower or "auto_mixed_precision" in lower:
            return (
                "accuracy-runtime-amp-autocast-drift",
                "accuracy",
                base_tags + ["accuracy", "amp", "autocast"],
                [],
                r"(amp\\.autocast|auto_mixed_precision)",
            )
        if "torch.frombuffer" in lower:
            return (
                "accuracy-operator-frombuffer-device-conversion-drift",
                "accuracy",
                base_tags + ["accuracy", "frombuffer", "device-conversion"],
                ["frombuffer"],
                r"torch\\.frombuffer",
            )
        if "np.histogram" in lower:
            return (
                "accuracy-operator-histogram-precision-drift",
                "accuracy",
                base_tags + ["accuracy", "histogram"],
                ["histogram"],
                r"np\\.histogram",
            )
        if "dsa" in lower and any(k in title for k in ["无法进行泛化", "参数都不能变动"]):
            return (
                "function-training-dsa-config-constraint",
                "failure",
                base_tags + ["training", "dsa", "config"],
                ["dsa"],
                r"(无法进行泛化|参数都不能变动)",
            )
        if "有一卡强制退出" in title:
            return (
                "failure-parallel-distributed-runtime",
                "failure",
                base_tags + ["parallel", "distributed"],
                [],
                r"(多副本|强制退出)",
            )
        if "tflops" in lower:
            return (
                "accuracy-training-tflops-metric-mismatch",
                "accuracy",
                base_tags + ["accuracy", "tflops", "metric"],
                [],
                r"TFLOPS|tflops",
            )
        if "模型推理" in title or "推理精度" in title or "精度不符合预期" in title:
            return (
                "accuracy-model-inference-regression",
                "accuracy",
                base_tags + ["accuracy", "model", "inference"],
                [],
                r"(模型推理|推理精度|精度不符合预期)",
            )
        if "inf" in lower:
            cause = "inf-semantics-drift"
        elif "nan" in lower or "异常值" in title:
            cause = "special-value-drift"
        elif any(k in lower for k in ["fp64", "fp16", "bf16", "float64", "float16", "bfloat16"]):
            cause = "dtype-drift"
        elif "反向" in title:
            cause = "backward-drift"
        else:
            cause = "precision-drift"
        op = operator or "runtime"
        return (
            f"accuracy-operator-{op}-{cause}" if operator else f"accuracy-runtime-{cause}",
            "accuracy",
            base_tags + ["accuracy", cause] + ([op] if operator else []),
            [op] if operator else [],
            r"(精度|异常值|不一致|nan|inf|loss|accuracy)",
        )

    if "mindformers" in lower:
        return (
            "function-mindformers-runtime-failure",
            "failure",
            base_tags + ["mindformers", "runtime"],
            [],
            r"MindFormers|mindformers",
        )

    if "用例失败" in title or "test_" in lower:
        return (
            "failure-test-general-case-failure",
            "failure",
            base_tags + ["testcase"],
            [],
            r"(用例失败|test_)",
        )

    return (
        "function-runtime-generic-failure",
        "failure",
        base_tags + ["runtime", "generic"],
        [],
        r"(报错|失败|error|failed|exception)",
    )


@dataclass
class IssueRef:
    issue_id: str
    title: str
    url: str
    project: str
    component: str
    backend: str
    priority: str
    status: str


@dataclass
class KnownIssueCard:
    card_id: str
    symptom: str
    tags: List[str] = field(default_factory=list)
    affects_operators: List[str] = field(default_factory=list)
    detection_pattern: str = ""
    occurrence_count: int = 0
    issue_refs: List[IssueRef] = field(default_factory=list)
    affects_platforms: List[str] = field(default_factory=list)
    severity: str = "low"
    lifecycle_state: str = "draft"
    fix_summary: str = ""
    fix_diff: str = ""


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def render_block(text: str, indent: int = 0) -> str:
    prefix = " " * indent
    lines = text.splitlines() or [""]
    return "\n".join(prefix + line for line in lines)


def infer_fix(card_id: str, symptom: str, operators: List[str]) -> Tuple[str, str]:
    if "accuracy-case-missing" in card_id or "test-adaptation-missing" in card_id:
        return (
            "补齐缺失或未适配的测试用例，使已知场景在门禁中可稳定复现并被及时拦截。",
            "1. 先确定缺的是精度用例、接口用例还是并行场景用例。\n2. 为最小复现路径补单测或系统测试。\n3. 对不稳定 case 固化随机种子、容差和环境前置条件。",
        )
    if "compiler-varargs-unsupported" in card_id:
        return (
            "为编译路径补齐 *args/**kwargs 展开支持，或在不支持场景增加前置拦截和明确错误提示。",
            "1. 还原触发编译失败的函数签名。\n2. 若目标场景应支持，补齐 varargs 解析与展开逻辑。\n3. 若暂不支持，在编译入口快速失败并补文档说明。",
        )
    if "compress-optimizer-config-invalid" in card_id or "zbv-config-invalid" in card_id or "memory-tracker-config-invalid" in card_id:
        return (
            "修正训练配置项与实现支持范围，避免不受支持组合在运行期才暴露失败。",
            "1. 明确配置项支持矩阵和互斥关系。\n2. 在启动前增加参数校验和报错说明。\n3. 对有效组合补回归测试，对无效组合补拦截测试。",
        )
    if "fusion-path-missing" in card_id:
        return (
            "修正图编译融合命中条件，确保目标 O1/融合路径可被稳定选择；不能命中时给出可解释日志。",
            "1. 对比期望融合路径与实际图编译日志。\n2. 补齐 pattern 匹配或图重写前置条件。\n3. 对未命中融合场景补诊断日志和回归 case。",
        )
    if "memory-oom" in card_id:
        return (
            "缩小显存/内存占用峰值，或在配置侧限制超规格组合，避免任务在运行中直接 OOM。",
            "1. 确认 OOM 出现在权重加载、图编译还是训练阶段。\n2. 通过切分、重算、降 batch 或调并行策略降低峰值。\n3. 对超规格配置增加前置校验。",
        )
    if "thread-sanitizer-race" in card_id:
        return (
            "修复共享状态并发访问与锁顺序问题，避免 TSAN 暴露 double lock / data race。",
            "1. 根据 TSAN 栈定位共享资源和锁顺序。\n2. 统一加锁顺序或改为无共享状态实现。\n3. 保留 TSAN 回归用例覆盖该路径。",
        )
    if "runtime-docstring-mismatch" in card_id or "error-message-mismatch" in card_id:
        return (
            "统一接口行为与对外文案，避免日志、注释或错误信息与真实实现不一致。",
            "1. 核对接口真实约束与当前文案。\n2. 修正文档、注释或错误信息模板。\n3. 对关键错误场景补字符串断言测试。",
        )
    if "metric-input-mismatch" in card_id or "interface-contract-mismatch" in card_id:
        return (
            "统一上下游接口契约，确保返回值结构、输入参数与调用方约定一致。",
            "1. 明确 producer 与 consumer 的张量/元组结构。\n2. 在边界处增加结构校验与转换。\n3. 为契约变更补回归用例，避免再次以运行时报错暴露。",
        )
    if "all-zero-output" in card_id:
        scope = operators[0] if operators else "算子"
        return (
            f"排查 {scope} 在特定 dtype/后端下的数值路径，修复输出被错误清零的问题。",
            "1. 固化最小输入并确认是前向全零还是中间值被截断。\n2. 对齐参考实现、dtype 提升和 cast 路径。\n3. 为该后端和 dtype 组合补精度回归用例。",
        )
    if "cann-upgrade-precision-drift" in card_id:
        return (
            "对比 CANN 升级前后的算子实现与库版本差异，定位精度回归是否由底层版本切换引入。",
            "1. 固化升级前后同一输入的输出差异。\n2. 对比相关算子选择路径、dtype cast 和底层库版本。\n3. 通过版本回退、白名单规避或兼容适配恢复精度基线。",
        )
    if "activation-update-state-drift" in card_id:
        return (
            "修正模块初始化后动态修改激活函数时的状态同步问题，保证执行图和模块真实状态一致。",
            "1. 检查修改激活函数后参数/缓存/图缓存是否同步更新。\n2. 避免旧图或旧状态继续参与执行。\n3. 为运行期改激活函数场景补回归用例。",
        )
    if "diffusers-scheduler-regression" in card_id:
        return (
            "对齐调度器参数更新公式与时间步语义，修复 diffusers 调度器迁移后的数值偏差。",
            "1. 逐步比对 scheduler 每一步的状态变量。\n2. 对齐参考框架的 timestep、sigma 和噪声更新逻辑。\n3. 为关键调度器补整链路精度回归测试。",
        )
    if "loss-spike" in card_id or "model-loss-drift" in card_id:
        return (
            "先定位首个偏离 step，再从并行策略、优化器状态和算子替换路径中确认训练 loss 漂移根因。",
            "1. 对齐种子、batch、权重和并行配置后比较 loss 曲线。\n2. 抓首个偏离 step 的关键张量和优化器状态。\n3. 按首个分歧层继续下钻到算子或通信路径。",
        )
    if "determinism-drift" in card_id:
        return (
            "收敛非确定性来源，保证同版本同配置重复运行结果可复现。",
            "1. 固化随机种子和通信顺序。\n2. 排查异步执行、原子更新和不稳定算子路径。\n3. 为确定性模式补重复运行一致性测试。",
        )
    if "aicpu-kernel-missing" in card_id:
        return (
            "补齐对应算子的 AICPU kernel 包装与发布物，或在不支持场景增加前置拦截。",
            "1. 确认 kernel 名称与 attr 配置是否正确。\n2. 检查发布包是否包含对应 aicpu kernel bin。\n3. 对缺失场景补明确错误提示和安装说明。",
        )
    if "workspace-failure" in card_id:
        op = operators[0] if operators else "operator"
        return (
            f"核对 {op} 对应的 CANN / ACLNN 版本与算子支持矩阵，优先通过升级环境或回退到受支持版本规避 WorkspaceSize 调用失败。",
            textwrap.dedent(
                f"""\
                1. 检查当前环境的 CANN 版本、驱动版本和算子支持列表。
                2. 若仅在升级后出现，优先回退到问题引入前版本，或升级到包含该 {op} 修复的版本。
                3. 若模型可接受替代实现，改用不依赖该 ACLNN 路径的算子/配置分支。
                """
            ).strip(),
        )
    if "module-import-missing" in card_id:
        return (
            "补齐缺失模块依赖，或在导入路径处增加环境/版本校验，避免运行时直接触发 ModuleNotFoundError。",
            "1. 确认运行环境已安装对应依赖包。\n2. 若依赖只在部分后端可用，增加显式版本/平台检查与友好报错。\n3. 对文档和启动脚本补充前置依赖说明。",
        )
    if "checkpoint-load-failure" in card_id:
        return (
            "统一 checkpoint / state_dict 的字段格式、文件类型和加载入口，必要时增加 BytesIO、safetensors 等输入形式的拦截或兼容分支。",
            "1. 校验 checkpoint 文件格式与加载接口契约。\n2. 对不支持的输入形式给出显式报错。\n3. 若存在权重字段命名差异，增加映射或转换脚本。",
        )
    if "mindir-export-load-failure" in card_id:
        return (
            "核对 MINDIR 导出/加载链路的场景约束，修正导出后端、加载后端和动态 shape/图模式的兼容条件。",
            "1. 先最小化复现导出和加载路径。\n2. 区分导出问题还是加载问题。\n3. 为不支持场景补前置拦截或文档说明。",
        )
    if "export-convert-map-missing" in card_id:
        return (
            "补齐导出算子到目标格式的 convert map，或者对暂不支持的算子增加明确拦截和替代路径。",
            "1. 定位缺失的 convert map key。\n2. 增加对应算子导出适配。\n3. 对未实现算子给清晰错误提示并更新支持列表。",
        )
    if "attribute-missing" in card_id:
        return (
            "统一接口导出与调用路径，保证运行时访问的属性在目标版本和目标后端下存在。",
            "1. 确认属性属于当前 API 契约。\n2. 若为版本差异，增加兼容分支或升级调用侧代码。\n3. 为缺失属性场景补充更明确的异常提示。",
        )
    if "not-callable" in card_id or "type-shape-mismatch" in card_id:
        return (
            "修正调用方式、参数类型或张量形状约束，使接口输入满足实际实现要求。",
            "1. 回看接口签名与调用参数。\n2. 在入口处增加参数校验与错误提示。\n3. 为异常参数场景补充回归用例，避免再次以运行时报错暴露。",
        )
    if "parallel-distributed" in card_id:
        return (
            "优先核对并行配置、通信组初始化、rank/world_size 参数和分布式 checkpoint 语义，必要时补充并行场景特化适配。",
            "1. 校验 world_size、rank、group name、并行配置项是否一致。\n2. 检查相关分布式接口在目标场景是否受支持。\n3. 对不兼容场景增加配置限制、显式报错或专门适配。",
        )
    if "hccl-group-config-invalid" in card_id or "msrun-config-invalid" in card_id:
        return (
            "修正启动参数和通信组配置，保证 msrun / HCCL 的路径、group name、rank 信息与实际集群环境一致。",
            "1. 校验启动脚本传参与实际文件路径。\n2. 检查 group name、rank、local rank 等分布式上下文。\n3. 在入口处增加无效配置拦截和文档说明。",
        )
    if "profiler" in card_id:
        return (
            "核对 profiler 开关、输出目录、采集阶段和离线解析流程，修正采集条件或补齐落盘逻辑。",
            "1. 检查 profiler 配置是否覆盖目标采集阶段。\n2. 校验输出目录、权限和多卡路径拼接。\n3. 为 FRAMEWORK/离线解析等关键落盘路径补回归用例。",
        )
    if "compile-graph" in card_id:
        return (
            "优先排查图编译路径、GE 初始化、内核选择和图模式特性交互，必要时降级到稳定配置或规避触发场景。",
            "1. 收集编译图和 GE 初始化日志。\n2. 缩小到最小复现图或具体特性交互。\n3. 通过关闭相关优化、回退版本或补图编译适配解决。",
        )
    if "mindformers-daily-case-failure" in card_id or "test-gate-case-failure" in card_id or "test-general-case-failure" in card_id:
        return (
            "先将问题缩小到具体失败用例，再按报错类型分别处理；门禁类问题优先补用例适配和稳定性防护。",
            "1. 先复现单个失败 case。\n2. 判断是环境波动、用例未适配还是产品逻辑回归。\n3. 修复后为同类门禁场景补充防护用例和稳定性校验。",
        )
    if "accuracy" in card_id:
        scope = operators[0] if operators else "当前场景"
        return (
            f"对 {scope} 相关精度问题，先做 baseline/dtype/tolerance 对齐，再缩小到算子或训练状态差异，最后按根因修复实现或用例。",
            "1. 先对齐输入、dtype、种子、容差和参考实现。\n2. 若是训练问题，定位 first divergence 或首个异常 step。\n3. 若是算子问题，补充最小复现并修正特殊值、dtype、反向或版本兼容逻辑。",
        )
    if "warning-log" in card_id or "doc-link-broken" in card_id:
        return (
            "修正文档、日志或非功能性输出问题，避免误导用户或影响工具链判断。",
            "1. 核对实际行为与日志/文档描述是否一致。\n2. 修改链接、文案或日志级别。\n3. 为对应非功能场景补检查项。",
        )
    if "performance" in card_id:
        return (
            "对性能类问题，先确认指标口径和复现配置，再排查版本、算子路径和并行策略导致的性能劣化。",
            "1. 固化 benchmark 配置和统计口径。\n2. 对比问题前后版本、算子开关和并行策略差异。\n3. 通过回退、替换实现或优化配置恢复性能基线。",
        )
    if "meta-device" in card_id:
        return (
            "为 meta device 场景补全显式支持或在不支持路径增加前置拦截，避免在运行到指针/算子阶段才崩溃。",
            "1. 判断目标算子是否应支持 meta。\n2. 不支持时在入口处快速失败并提示。\n3. 支持时补齐 meta 推理路径和相关测试。",
        )
    if "warning-log" in card_id:
        return (
            "降低无效 warning 噪音或补充日志等级过滤，避免影响任务侧判断与告警收敛。",
            "1. 确认 warning 是否为预期行为。\n2. 预期场景下调整日志等级或去重输出。\n3. 非预期场景下修复触发条件。",
        )
    return (
        "先将问题缩小到具体触发条件，再补充输入校验、环境校验或实现兼容逻辑，避免继续落到宽泛运行时失败分类。",
        "1. 从 issue_refs 中挑选代表性 case 做最小复现。\n2. 根据真实报错补更细的 known_issue 卡。\n3. 为修复点补回归用例并收敛检测规则。",
    )


def write_yaml(card: KnownIssueCard, path: Path) -> None:
    titles = [ref.title for ref in card.issue_refs[:5]]
    components = sorted({ref.component for ref in card.issue_refs if ref.component})
    backends = sorted({ref.backend for ref in card.issue_refs if ref.backend})
    description = textwrap.dedent(
        f"""\
        Imported from gitcode问题单26.1-26.4.xlsx using title-based grouping.
        This card groups {card.occurrence_count} issue(s) with similar signatures.
        Representative titles:
        """
    )
    for t in titles:
        description += f"- {t}\n"
    if components:
        description += f"Components: {', '.join(components)}\n"
    if backends:
        description += f"Backends: {', '.join(backends)}\n"

    lines = [
        "kind: known_issue",
        f"id: {card.card_id}",
        f"symptom: {card.symptom}",
        f"severity: {card.severity}",
        "lifecycle:",
        f"  state: {card.lifecycle_state}",
        "source:",
        "  kind: manual",
        "confidence:",
        "  level: observed",
        f"tags: [{', '.join(dedupe_keep_order(card.tags))}]",
        f"occurrence_count: {card.occurrence_count}",
    ]
    if card.affects_operators:
        lines.append(f"affects_operators: [{', '.join(dedupe_keep_order(card.affects_operators))}]")
    if card.affects_platforms:
        lines.append(f"affects_platforms: [{', '.join(dedupe_keep_order(card.affects_platforms))}]")
    lines.extend(
        [
            "detection:",
            f'  pattern: "{card.detection_pattern}"',
            "description: |",
            render_block(description.rstrip(), 2),
            "issue_refs:",
        ]
    )
    for ref in card.issue_refs:
        lines.extend(
            [
                f'  - issue_id: "{ref.issue_id}"',
                f'    title: "{ref.title.replace(chr(34), chr(39))}"',
                f'    url: "{ref.url}"',
                f'    project: "{ref.project}"',
                f'    component: "{ref.component}"',
                f'    backend: "{ref.backend}"',
                f'    priority: "{ref.priority}"',
                f'    status: "{ref.status}"',
            ]
        )
    lines.extend(
        [
            "fix:",
            f'  summary: "{card.fix_summary.replace(chr(34), chr(39))}"',
            "  diff: |",
            render_block(card.fix_diff.rstrip(), 4),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_issue_id(url: str) -> str:
    m = re.search(r"/issues/(\d+)", url or "")
    return m.group(1) if m else ""


def build_cards(rows: List[Dict[str, str]]) -> Dict[str, KnownIssueCard]:
    cards: Dict[str, KnownIssueCard] = {}
    for row in rows:
        title = row.get("标题", "")
        if not title or title.startswith("CVE-"):
            continue

        card_id, symptom, tags, operators, pattern = classify_issue(
            title,
            row.get("关联组件", ""),
            row.get("所属项目", ""),
        )
        ref = IssueRef(
            issue_id=parse_issue_id(row.get("issue链接", "")),
            title=title,
            url=row.get("issue链接", ""),
            project=row.get("所属项目", ""),
            component=row.get("关联组件", ""),
            backend=row.get("问题后端类型", ""),
            priority=row.get("issue优先级", ""),
            status=row.get("状态", ""),
        )
        card = cards.get(card_id)
        if card is None:
            fix_summary, fix_diff = infer_fix(card_id, symptom, operators)
            card = KnownIssueCard(
                card_id=card_id,
                symptom=symptom,
                tags=tags,
                affects_operators=operators,
                detection_pattern=pattern,
                fix_summary=fix_summary,
                fix_diff=fix_diff,
            )
            cards[card_id] = card
        card.issue_refs.append(ref)
        card.occurrence_count += 1
        card.tags.extend(tags)
        card.affects_operators.extend(operators)
        card.affects_platforms.extend(normalize_platforms(row.get("问题后端类型", "")))
        severity = ISSUE_PRIORITY_MAP.get(row.get("issue优先级", ""), "low")
        if SEVERITY_ORDER[severity] > SEVERITY_ORDER[card.severity]:
            card.severity = severity
        if row.get("issue状态", "").upper() != "DONE" or row.get("状态", "").lower() != "closed":
            card.lifecycle_state = "draft"

    for card in cards.values():
        card.tags = dedupe_keep_order(card.tags)
        card.affects_operators = dedupe_keep_order(card.affects_operators)
        card.affects_platforms = dedupe_keep_order(card.affects_platforms)
        if card.occurrence_count >= 3 and card.lifecycle_state == "draft":
            # keep draft; we only imported signatures, not verified fixes
            pass
    return cards


def rewrite_pack(generated_cards: Dict[str, KnownIssueCard]) -> None:
    text = PACK_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    preserved = []
    for line in lines:
        if "kind: known_issue" in line:
            continue
        preserved.append(line)

    manual_entries = [
        f"  - {{ id: {card_id}, kind: known_issue, path: {path} }}"
        for card_id, path in MANUAL_KNOWN_ISSUES.items()
    ]
    generated_entries = [
        f"  - {{ id: {card_id}, kind: known_issue, path: known_issues/generated/{card_id}.yaml }}"
        for card_id in sorted(generated_cards)
    ]

    out = []
    for line in preserved:
        if re.match(r"\s*known_issues:\s*\d+", line):
            total = len(manual_entries) + len(generated_entries)
            out.append(re.sub(r"\d+", str(total), line, count=1))
            continue
        out.append(line)
        if line.strip() == "- { id: sdpa, kind: operator, path: operators/sdpa.yaml }":
            out.extend(manual_entries)
            out.extend(generated_entries)
    PACK_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", default=str(XLSX_PATH))
    args = parser.parse_args()

    rows = read_sheet_rows(Path(args.xlsx))
    # Skip first two data rows in the spreadsheet: CVE tracking entries are intentionally excluded.
    rows = rows[2:]

    cards = build_cards(rows)
    for manual_id in MANUAL_KNOWN_ISSUES:
        cards.pop(manual_id, None)
    KNOWN_ISSUES_DIR.mkdir(parents=True, exist_ok=True)
    for old in KNOWN_ISSUES_DIR.glob("*.yaml"):
        old.unlink()
    for card_id, card in sorted(cards.items()):
        write_yaml(card, KNOWN_ISSUES_DIR / f"{card_id}.yaml")
    rewrite_pack(cards)

    print(f"generated_cards={len(cards)}")
    for card_id, card in sorted(cards.items(), key=lambda x: (-x[1].occurrence_count, x[0]))[:20]:
        print(f"{card.occurrence_count:3d} {card_id}")


if __name__ == "__main__":
    main()
