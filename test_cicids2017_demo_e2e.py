#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_cicids2017_demo_e2e.py  (v2 — 使用 cicids2017_demo_v2 PCAP + ExtendedAnomalyDetector)

端對端測試：將 cicids2017_demo_v2/ 中的 10 個 PCAP 分別餵入
unsup_cicids2017 和 semi_cicids2017 兩種模型，驗證：
  1. 分析流程是否正常完成（無錯誤）
  2. 正常流量的異常率是否合理（期望 < 30%）
  3. 攻擊流量的異常率是否合理（期望 > 50%）
  4. 規則式偵測引擎是否觸發正確歸因的告警（src_ip=攻擊者，非受害者）
  5. Grad-CAM 影像是否產生

用法：python test_cicids2017_demo_e2e.py
      python test_cicids2017_demo_e2e.py --pcap-dir test_pcaps/cicids2017_demo_v2
"""

import os
import sys
import json
import time
import shutil
import argparse
import traceback

# Django setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "network_platform.settings")

import django
django.setup()

from django.conf import settings
from analyzer.models import AnalysisSession, CNNResult, Alert, GradCAMImage

CORE_DIR = os.path.join(BASE_DIR, "core")
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

ATTACKER_IP = "172.16.0.1"   # cicids2017_demo_v2 拓樸中的攻擊者 IP
MODELS = ["unsup_cicids2017", "semi_cicids2017"]

# 每個攻擊 PCAP 期望出現的告警關鍵字（只要 attack_type 包含即可）
EXPECTED_ALERT = {
    "attack_01_ddos.pcap":           "SYN Flood",
    "attack_02_portscan.pcap":       "Port Scan",
    "attack_03_ssh_bruteforce.pcap": "SSH Brute Force",
    "attack_04_web_bruteforce.pcap": "Web Login Brute Force",
    "attack_05_dos_hulk.pcap":       "HTTP Flood",
}


def run_analysis(pcap_abs_path, model_key):
    """執行分析並回傳結果字典。"""
    from analyzer.tasks import execute_pcap_analysis

    fname = os.path.basename(pcap_abs_path)
    upload_rel = os.path.join("uploads", f"e2e_v2_{model_key}_{fname}")
    upload_abs = os.path.join(str(settings.MEDIA_ROOT), upload_rel)
    os.makedirs(os.path.dirname(upload_abs), exist_ok=True)
    shutil.copy2(pcap_abs_path, upload_abs)

    session = AnalysisSession.objects.create(
        mode="pcap",
        label=f"E2E-v2: {fname} ({model_key})",
        pcap_file=upload_rel,
        model_key=model_key,
        task_status="pending",
    )

    try:
        execute_pcap_analysis(session.id, auto_gradcam=True)
        session.refresh_from_db()

        cnn = None
        try:
            cnn = CNNResult.objects.get(session=session)
        except CNNResult.DoesNotExist:
            pass

        alerts = list(Alert.objects.filter(session=session).values(
            "attack_type", "severity", "src_ip", "detail"
        ))
        gradcam_total = GradCAMImage.objects.filter(session=session).count()
        gradcam_anomaly = GradCAMImage.objects.filter(session=session, is_anomaly=True).count()
        gradcam_normal = GradCAMImage.objects.filter(session=session, is_anomaly=False).count()

        total = cnn.normal_count + cnn.anomaly_count if cnn else 0
        anomaly_rate = cnn.anomaly_count / max(total, 1) if cnn else 0.0

        return {
            "task_status": session.task_status,
            "error": session.error_message or None,
            "total_samples": total,
            "normal_count": cnn.normal_count if cnn else 0,
            "anomaly_count": cnn.anomaly_count if cnn else 0,
            "anomaly_rate": anomaly_rate,
            "detection_rate": cnn.detection_rate if cnn else 0,
            "threshold": cnn.threshold if cnn else None,
            "avg_normal_error": cnn.avg_normal_error if cnn else 0,
            "avg_attack_error": cnn.avg_attack_error if cnn else 0,
            "known_attack_count": cnn.known_attack_count if cnn else 0,
            "unknown_attack_count": cnn.unknown_attack_count if cnn else 0,
            "alerts": alerts,
            "alert_count": len(alerts),
            "gradcam_total": gradcam_total,
            "gradcam_anomaly": gradcam_anomaly,
            "gradcam_normal": gradcam_normal,
            "packet_count": session.packet_count,
            "session_id": session.id,
        }
    except Exception as e:
        session.refresh_from_db()
        return {
            "task_status": session.task_status,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "total_samples": 0,
            "normal_count": 0,
            "anomaly_count": 0,
            "anomaly_rate": 0,
            "detection_rate": 0,
            "threshold": None,
            "avg_normal_error": 0,
            "avg_attack_error": 0,
            "known_attack_count": 0,
            "unknown_attack_count": 0,
            "alerts": [],
            "alert_count": 0,
            "gradcam_total": 0,
            "gradcam_anomaly": 0,
            "gradcam_normal": 0,
            "packet_count": 0,
            "session_id": session.id,
        }


def _verdict_normal(res, fname):
    """判定正常流量案例。"""
    if res["task_status"] == "failed" or res.get("error"):
        return "FAIL", [f"錯誤: {res.get('error', 'unknown')}"]
    problems = []
    if res["total_samples"] == 0:
        return "FAIL", ["無 CNN 樣本（Flow 轉換失敗）"]
    if res["total_samples"] < 3:
        problems.append(f"樣本數過少 ({res['total_samples']})，統計無意義")
    if res["anomaly_rate"] > 0.3:
        problems.append(f"誤報率 {res['anomaly_rate']:.1%} > 30%")
    # 正常 PCAP 不應有告警
    victim_alerts = [a for a in res["alerts"] if a["src_ip"] == ATTACKER_IP]
    if victim_alerts:
        problems.append(f"正常流量觸發 {len(victim_alerts)} 個告警")
    if problems:
        return "WARN", problems
    return "PASS", []


def _verdict_attack(res, fname):
    """判定攻擊流量案例（加入告警歸因驗證）。"""
    if res["task_status"] == "failed" or res.get("error"):
        return "FAIL", [f"錯誤: {res.get('error', 'unknown')}"]
    problems = []
    if res["total_samples"] == 0:
        return "FAIL", ["無 CNN 樣本（Flow 轉換失敗）"]
    if res["anomaly_rate"] < 0.5:
        problems.append(f"攻擊流量異常率 {res['anomaly_rate']:.1%} < 50%")

    # 告警驗證
    expected_kw = EXPECTED_ALERT.get(fname)
    if expected_kw:
        matching = [a for a in res["alerts"] if expected_kw in a["attack_type"]]
        correct_src = [a for a in matching if a["src_ip"] == ATTACKER_IP]
        victim_misattrib = [a for a in res["alerts"] if a["src_ip"] != ATTACKER_IP]
        if not matching:
            problems.append(f"缺少期望告警「{expected_kw}」")
        elif not correct_src:
            problems.append(f"告警「{expected_kw}」的 src_ip 錯誤（期望 {ATTACKER_IP}）")
        if victim_misattrib:
            problems.append(f"誤將受害者 ({victim_misattrib[0]['src_ip']}) 列為攻擊來源")

    if problems:
        return "WARN", problems
    return "PASS", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap-dir", default=os.path.join(BASE_DIR, "test_pcaps", "cicids2017_demo_v2"))
    args = ap.parse_args()

    demo_dir = args.pcap_dir
    if not os.path.exists(demo_dir):
        print(f"[ERROR] PCAP 目錄不存在: {demo_dir}")
        return

    normal_pcaps = sorted([f for f in os.listdir(demo_dir) if f.startswith("normal_") and f.endswith(".pcap")])
    attack_pcaps = sorted([f for f in os.listdir(demo_dir) if f.startswith("attack_") and f.endswith(".pcap")])

    print("=" * 72)
    print("  CICIDS2017 Demo PCAP E2E Test  (v2 — ExtendedAnomalyDetector)")
    print(f"  Normal PCAPs : {len(normal_pcaps)}")
    print(f"  Attack PCAPs : {len(attack_pcaps)}")
    print(f"  Models       : {', '.join(MODELS)}")
    print(f"  PCAP Dir     : {demo_dir}")
    print("=" * 72)

    all_results = {}
    total_tests = pass_count = warn_count = fail_count = 0

    for model_key in MODELS:
        print(f"\n{'=' * 72}")
        print(f"  Model: {model_key}")
        print(f"{'=' * 72}")
        model_results = {}

        for is_attack, pcap_list in [(False, normal_pcaps), (True, attack_pcaps)]:
            tag = "Attack" if is_attack else "Normal"
            print(f"\n  --- {tag} Traffic ---")
            for fname in pcap_list:
                total_tests += 1
                pcap_path = os.path.join(demo_dir, fname)
                label = fname.replace(".pcap", "")
                print(f"\n  [{label}]")

                t0 = time.time()
                res = run_analysis(pcap_path, model_key)
                elapsed = time.time() - t0

                if is_attack:
                    verdict, problems = _verdict_attack(res, fname)
                else:
                    verdict, problems = _verdict_normal(res, fname)

                if verdict == "PASS":
                    pass_count += 1
                elif verdict == "WARN":
                    warn_count += 1
                else:
                    fail_count += 1

                # 印出詳細資訊
                print(f"    task_status  : {res['task_status']}")
                print(f"    packet_count : {res['packet_count']} (rules engine)")
                print(f"    total_samples: {res['total_samples']} (CNN/VAE flow)")
                print(f"    normal/anom  : {res['normal_count']} / {res['anomaly_count']} ({res['anomaly_rate']:.1%})")
                if res["threshold"] is not None:
                    print(f"    threshold    : {res['threshold']:.8g}")
                    if res["total_samples"] > 0:
                        ratio = res["avg_attack_error"] / res["threshold"] if res["avg_attack_error"] else 0
                        print(f"    avg_error    : normal={res['avg_normal_error']:.6f}  "
                              f"attack={res['avg_attack_error']:.6f}  ({ratio:.1f}x threshold)")
                print(f"    alerts       : {res['alert_count']}")
                for a in res["alerts"][:5]:
                    print(f"      [{a['severity']}] {a['attack_type']} from {a['src_ip']}")
                if res["alert_count"] > 5:
                    print(f"      ... and {res['alert_count']-5} more")
                print(f"    gradcam      : {res['gradcam_total']} (anomaly={res['gradcam_anomaly']}, normal={res['gradcam_normal']})")
                if res.get("known_attack_count") or res.get("unknown_attack_count"):
                    print(f"    hybrid       : known={res['known_attack_count']}, unknown={res['unknown_attack_count']}")
                for p in problems:
                    print(f"    [!] {p}")
                print(f"    verdict      : {verdict}  (elapsed: {elapsed:.1f}s)")

                model_results[fname] = {**res, "verdict": verdict,
                                         "expected": "attack" if is_attack else "normal",
                                         "elapsed": round(elapsed, 1)}

        all_results[model_key] = model_results

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print(f"  Total : {total_tests}")
    print(f"  PASS  : {pass_count}")
    print(f"  WARN  : {warn_count}")
    print(f"  FAIL  : {fail_count}")
    print("=" * 72)

    for model_key, model_results in all_results.items():
        nr = [d["anomaly_rate"] for d in model_results.values()
              if d["expected"] == "normal" and d["total_samples"] >= 3]
        ar = [d["anomaly_rate"] for d in model_results.values()
              if d["expected"] == "attack" and d["total_samples"] > 0]
        print(f"\n  {model_key}:")
        print(f"    Avg normal anomaly rate : {sum(nr)/len(nr):.1%}" if nr else "    Normal: no valid samples")
        print(f"    Avg attack anomaly rate : {sum(ar)/len(ar):.1%}" if ar else "    Attack: no valid samples")

    # Save
    out_path = os.path.join(demo_dir, "e2e_test_results_v2.json")
    json_results = {}
    for mk, mr in all_results.items():
        json_results[mk] = {}
        for fn, data in mr.items():
            serializable = {k: v for k, v in data.items() if k != "traceback"}
            try:
                json.dumps(serializable)
            except (TypeError, ValueError):
                serializable = {k: str(v) for k, v in serializable.items()}
            json_results[mk][fn] = serializable
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
