#!/usr/bin/env python3
# ============================================================
# core/tests/run_gradcam_tests.py  - Grad-CAM 測試執行器
# ============================================================
"""
執行所有 Grad-CAM 測試，輸出分類彙整報告。

使用方式：
  # 執行所有測試（從 core/ 目錄）
  python tests/run_gradcam_tests.py

  # 執行單一模組
  python tests/run_gradcam_tests.py --module hooks

  # 快速模式（跳過 scorecam 等慢速測試）
  python tests/run_gradcam_tests.py --fast

  # 顯示詳細輸出
  python tests/run_gradcam_tests.py -v

  # 也可以直接用 unittest 模組執行
  python -m unittest tests/test_gradcam_hooks.py -v
  python -m unittest discover -s tests -p "test_gradcam*.py" -v
"""

import sys
import os
import time
import argparse
import unittest

# 確保 core/ 與 core/tests/ 都在 sys.path
_CORE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TESTS = os.path.dirname(os.path.abspath(__file__))
for _d in [_CORE, _TESTS]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


# ── 測試模組清單 ──────────────────────────────────────────────
TEST_MODULES = {
    "hooks":       ("tests.test_gradcam_hooks",       "HookManager 單元測試"),
    "core":        ("tests.test_gradcam_core",        "GradCAM 核心演算法"),
    "visualizer":  ("tests.test_gradcam_visualizer",  "視覺化模組"),
    "analyzer":    ("tests.test_gradcam_analyzer",    "分析器與報告匯出"),
    "integration": ("tests.test_gradcam_integration", "端到端整合測試"),
}

# 快速模式排除的測試（含 scorecam 等較慢的測試）
FAST_SKIP_PATTERNS = [
    "test_scorecam",
    "test_large_batch",
]


class TimedTextTestResult(unittest.TextTestResult):
    """加入每個測試的執行時間顯示。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_times = {}

    def startTest(self, test):
        self._start_times[test] = time.time()
        super().startTest(test)

    def stopTest(self, test):
        elapsed = time.time() - self._start_times.get(test, time.time())
        if self.showAll and elapsed > 0.5:
            self.stream.write(f"  ⏱  {elapsed:.2f}s\n")
        super().stopTest(test)


class GradCAMTestRunner:
    def __init__(self, verbosity: int = 2, fast_mode: bool = False):
        self.verbosity = verbosity
        self.fast_mode = fast_mode

    def _load_suite(self, module_key: str) -> unittest.TestSuite:
        module_name, _ = TEST_MODULES[module_key]
        loader = unittest.TestLoader()
        suite  = loader.loadTestsFromName(module_name)

        if self.fast_mode:
            filtered = unittest.TestSuite()
            for test in self._flatten(suite):
                if not any(p in test.id() for p in FAST_SKIP_PATTERNS):
                    filtered.addTest(test)
            return filtered
        return suite

    def _flatten(self, suite):
        for item in suite:
            if isinstance(item, unittest.TestSuite):
                yield from self._flatten(item)
            else:
                yield item

    def run_all(self, modules=None) -> bool:
        targets = modules or list(TEST_MODULES.keys())
        all_passed = True
        grand_total = grand_fail = grand_error = grand_skip = 0
        grand_start = time.time()

        print("\n" + "═" * 65)
        print("  Grad-CAM 測試套件")
        print(f"  模式：{'快速（跳過慢速測試）' if self.fast_mode else '完整'}")
        print("═" * 65)

        module_results = []

        for key in targets:
            if key not in TEST_MODULES:
                print(f"  [跳過] 未知模組：{key}")
                continue

            _, desc = TEST_MODULES[key]
            print(f"\n▶  {desc}")
            print("─" * 55)

            t0    = time.time()
            suite = self._load_suite(key)
            n_tests = suite.countTestCases()

            if n_tests == 0:
                print("  （無測試案例）")
                continue

            buf = unittest.runner._WritelnDecorator(sys.stdout)
            result = unittest.TextTestResult(buf, descriptions=True,
                                              verbosity=self.verbosity)
            suite.run(result)
            elapsed = time.time() - t0

            passed = n_tests - len(result.failures) - len(result.errors) - len(result.skipped)
            status = "✅ PASS" if result.wasSuccessful() else "❌ FAIL"

            print(f"\n  {status}  通過 {passed}/{n_tests}  "
                  f"失敗 {len(result.failures)}  "
                  f"錯誤 {len(result.errors)}  "
                  f"跳過 {len(result.skipped)}  "
                  f"({elapsed:.1f}s)")

            grand_total += n_tests
            grand_fail  += len(result.failures)
            grand_error += len(result.errors)
            grand_skip  += len(result.skipped)

            if not result.wasSuccessful():
                all_passed = False
                for test, msg in result.failures + result.errors:
                    print(f"\n  ⚠  {test.id()}")
                    # 只顯示最後 5 行錯誤訊息
                    lines = msg.strip().split("\n")
                    for line in lines[-5:]:
                        print(f"     {line}")

            module_results.append((key, desc, result.wasSuccessful(), passed, n_tests))

        # ── 總結 ────────────────────────────────────────────
        total_elapsed = time.time() - grand_start
        grand_passed  = grand_total - grand_fail - grand_error - grand_skip

        print("\n" + "═" * 65)
        print(f"  測試完成  總耗時：{total_elapsed:.1f}s")
        print(f"  總計：{grand_total} 個測試案例")
        print(f"  ✅ 通過：{grand_passed}   ❌ 失敗：{grand_fail}   "
              f"⚠ 錯誤：{grand_error}   ⏭ 跳過：{grand_skip}")

        print("\n  模組摘要：")
        for key, desc, ok, passed, total in module_results:
            icon = "✅" if ok else "❌"
            print(f"    {icon}  {desc:<35} {passed}/{total}")

        if all_passed:
            print("\n  🎉 所有測試通過！Grad-CAM 模組功能正常。")
        else:
            print("\n  ⚠  有測試失敗，請檢查上方錯誤訊息。")

        print("═" * 65 + "\n")
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM 測試執行器")
    parser.add_argument("--module", "-m",
                        nargs="+",
                        choices=list(TEST_MODULES.keys()) + ["all"],
                        default=["all"],
                        help="要執行的測試模組（預設: all）")
    parser.add_argument("--fast",   action="store_true",
                        help="快速模式：跳過 scorecam 等較慢的測試")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="顯示詳細輸出")
    args = parser.parse_args()

    modules = None if "all" in args.module else args.module
    runner  = GradCAMTestRunner(
        verbosity = 2 if args.verbose else 1,
        fast_mode = args.fast,
    )
    success = runner.run_all(modules)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
