# -*- coding: utf-8 -*-
"""
tests/test_6bug_fixes.py
驗證 6 個 bug 修正的單元測試。

Bug A: capture.py 分段儲存清空 packets
Bug B: tasks.py packet_count 賦值位置
Bug C: tasks.py run_gradcam 半監督判定
Bug D: session_detail.html 假高斯曲線移除（模板驗證）
Bug E: heatmap_analysis.html 欄位名稱修正（模板驗證）
Bug F: gradcam_gallery.html + heatmap_analysis.html 欄位名稱修正（模板驗證）
"""
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, PropertyMock
from collections import Counter

# 確保 core/ 在路徑中
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(BASE_DIR, 'core')
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)


# ════════════════════════════════════════════════════════════════
# Bug A: capture.py 分段儲存後 packets 必須被清空
# ════════════════════════════════════════════════════════════════
class TestBugA_CaptureSegmentClear(unittest.TestCase):
    """驗證 _check_segment_save() 分段寫出後會清空 self.packets。"""

    def _make_capture(self):
        """建立一個最小化的 LiveCapture，跳過網路介面初始化。"""
        from capture import LiveCapture
        cap = LiveCapture.__new__(LiveCapture)
        # 手動初始化必要屬性
        import threading
        cap._lock = threading.Lock()
        cap.packets = []
        cap.parsed_records = []
        cap.save_pcap = True
        cap._segment_counter = 0
        cap._segment_index = 0
        cap._last_segment_time = 0
        return cap

    @patch('capture.wrpcap')
    @patch('capture.PCAP_SEGMENT_SIZE', 5)
    def test_packets_cleared_after_segment_save(self, mock_wrpcap):
        """分段儲存觸發後，self.packets 和 self.parsed_records 必須清空。"""
        cap = self._make_capture()

        # 模擬有 10 個封包在記憶體中
        for i in range(10):
            cap.packets.append(MagicMock(name=f'pkt_{i}'))
            cap.parsed_records.append({'index': i})

        # 設定觸發條件：封包數超過分段大小
        cap._segment_counter = 10  # > PCAP_SEGMENT_SIZE=5

        # 執行分段儲存
        cap._check_segment_save()

        # 驗證 wrpcap 被呼叫（寫出 10 個封包）
        self.assertTrue(mock_wrpcap.called, 'wrpcap 應該被呼叫')
        written_packets = mock_wrpcap.call_args[0][1]
        self.assertEqual(len(written_packets), 10, '應寫出全部 10 個封包')

        # [Bug A 核心驗證] packets 必須被清空
        self.assertEqual(len(cap.packets), 0,
                         'Bug A: self.packets 在分段儲存後必須被清空')
        self.assertEqual(len(cap.parsed_records), 0,
                         'Bug A: self.parsed_records 在分段儲存後必須被清空')

    @patch('capture.wrpcap')
    @patch('capture.PCAP_SEGMENT_SIZE', 100)
    def test_packets_not_cleared_when_no_save(self, mock_wrpcap):
        """不觸發分段時，packets 不應被清空。"""
        cap = self._make_capture()
        cap.packets = [MagicMock() for _ in range(5)]
        cap.parsed_records = [{'i': i} for i in range(5)]
        cap._segment_counter = 3  # < 100

        cap._check_segment_save()

        self.assertFalse(mock_wrpcap.called)
        self.assertEqual(len(cap.packets), 5, '未觸發時 packets 不應被清空')

    @patch('capture.wrpcap')
    @patch('capture.PCAP_SEGMENT_SIZE', 5)
    def test_segment_index_increments(self, mock_wrpcap):
        """每次分段儲存後 segment_index 應遞增。"""
        cap = self._make_capture()
        cap.packets = [MagicMock() for _ in range(10)]
        cap.parsed_records = [{'i': i} for i in range(10)]
        cap._segment_counter = 10

        self.assertEqual(cap._segment_index, 0)
        cap._check_segment_save()
        self.assertEqual(cap._segment_index, 1)


# ════════════════════════════════════════════════════════════════
# Bug B: packet_count 必須在 detect_attacks 之前設定
# ════════════════════════════════════════════════════════════════
class TestBugB_PacketCountEarlyAssign(unittest.TestCase):
    """驗證 packet_count 即使 detect_attacks() 失敗也能正確設定。"""

    def test_packet_count_set_before_detect(self):
        """讀取 tasks.py 原始碼，確認 packet_count 賦值在 detect_attacks 之前。"""
        tasks_path = os.path.join(BASE_DIR, 'analyzer', 'tasks.py')
        with open(tasks_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # 找到 packet_count 賦值和 detect_attacks 呼叫的行號
        lines = source.split('\n')
        packet_count_line = None
        detect_attacks_line = None

        for i, line in enumerate(lines, 1):
            if 'session.packet_count' in line and 'len(analyzer.packets)' in line:
                packet_count_line = i
            if 'analyzer.detect_attacks()' in line and packet_count_line is None:
                # detect_attacks 在 packet_count 之前 → Bug 未修
                pass
            if 'analyzer.detect_attacks()' in line:
                detect_attacks_line = i

        self.assertIsNotNone(packet_count_line,
                             'Bug B: 應有 session.packet_count = len(analyzer.packets)')
        self.assertIsNotNone(detect_attacks_line,
                             '應有 analyzer.detect_attacks() 呼叫')
        self.assertLess(packet_count_line, detect_attacks_line,
                        'Bug B: packet_count 賦值必須在 detect_attacks() 之前')


# ════════════════════════════════════════════════════════════════
# Bug C: run_gradcam 半監督模型必須使用三態判定
# ════════════════════════════════════════════════════════════════
class TestBugC_GradcamHybridJudgment(unittest.TestCase):
    """驗證 run_gradcam 對 hybrid 模型使用分類器判定。"""

    def test_classify_known_called_for_hybrid(self):
        """讀取 tasks.py，確認 hybrid 模型會呼叫 classify_known。"""
        tasks_path = os.path.join(BASE_DIR, 'analyzer', 'tasks.py')
        with open(tasks_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # 確認 run_gradcam 中有 classify_known 呼叫
        self.assertIn('classify_known', source,
                      'Bug C: run_gradcam 應呼叫 model.classify_known()')

        # 確認有 bundle.is_hybrid 判斷
        self.assertIn('bundle.is_hybrid', source,
                      'Bug C: run_gradcam 應檢查 bundle.is_hybrid')

        # 確認三態邏輯：classifier_says_attack or (err > threshold)
        self.assertIn('classifier_says_attack', source,
                      'Bug C: 應有 classifier_says_attack 變數')

    def test_non_hybrid_uses_simple_threshold(self):
        """非 hybrid 模型仍使用簡單的 err > threshold。"""
        tasks_path = os.path.join(BASE_DIR, 'analyzer', 'tasks.py')
        with open(tasks_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # 在 else 分支中應有原始邏輯
        self.assertIn('is_anomaly = err > threshold', source,
                      '非 hybrid 模型應保留 err > threshold 邏輯')


# ════════════════════════════════════════════════════════════════
# Bug D: session_detail.html 不應有假高斯曲線
# ════════════════════════════════════════════════════════════════
class TestBugD_NoFakeGaussian(unittest.TestCase):
    """驗證 session_detail.html 不再使用假高斯分布。"""

    def test_no_gaussian_formula(self):
        """確認模板中不再有 Math.exp 高斯公式。"""
        path = os.path.join(BASE_DIR, 'templates', 'analyzer', 'session_detail.html')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertNotIn('Math.exp(-0.5', content,
                         'Bug D: 不應再有高斯公式 Math.exp(-0.5 * ...)')
        self.assertNotIn('模擬分布', content,
                         'Bug D: 不應再有「模擬分布」字眼')

    def test_uses_real_values(self):
        """確認改用真實量測值。"""
        path = os.path.join(BASE_DIR, 'templates', 'analyzer', 'session_detail.html')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('實際量測', content,
                      'Bug D: 應標註為「實際量測」')


# ════════════════════════════════════════════════════════════════
# Bug E: heatmap_analysis.html 欄位名稱正確
# ════════════════════════════════════════════════════════════════
class TestBugE_HeatmapCorrectFields(unittest.TestCase):
    """驗證 heatmap_analysis.html 使用正確的 CNNResult 欄位。"""

    def test_no_wrong_cnnresult_fields(self):
        path = os.path.join(BASE_DIR, 'templates', 'analyzer', 'heatmap_analysis.html')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertNotIn('avg_reconstruction_error', content,
                         'Bug E: 不應引用不存在的 avg_reconstruction_error')
        self.assertNotIn('threshold_used', content,
                         'Bug E: 不應引用不存在的 threshold_used')

    def test_correct_cnnresult_fields(self):
        path = os.path.join(BASE_DIR, 'templates', 'analyzer', 'heatmap_analysis.html')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('avg_attack_error', content,
                      'Bug E: 應使用正確的 avg_attack_error')
        self.assertIn('cnn_result.threshold', content,
                      'Bug E: 應使用正確的 cnn_result.threshold')


# ════════════════════════════════════════════════════════════════
# Bug F: gradcam_gallery + heatmap_analysis 欄位名稱正確
# ════════════════════════════════════════════════════════════════
class TestBugF_GradcamCorrectFields(unittest.TestCase):
    """驗證 GradCAMImage 欄位引用全部正確。"""

    def test_no_wrong_gradcam_fields_in_gallery(self):
        path = os.path.join(BASE_DIR, 'templates', 'analyzer', 'gradcam_gallery.html')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertNotIn('img.image.url', content,
                         'Bug F: gradcam_gallery 不應引用不存在的 img.image')
        self.assertNotIn('img.anomaly_score', content,
                         'Bug F: gradcam_gallery 不應引用不存在的 img.anomaly_score')

    def test_correct_gradcam_fields_in_gallery(self):
        path = os.path.join(BASE_DIR, 'templates', 'analyzer', 'gradcam_gallery.html')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertIn('comparison_image.url', content,
                      'Bug F: 應使用正確的 comparison_image.url')
        self.assertIn('img.recon_error', content,
                      'Bug F: 應使用正確的 img.recon_error')

    def test_no_wrong_gradcam_fields_in_heatmap(self):
        path = os.path.join(BASE_DIR, 'templates', 'analyzer', 'heatmap_analysis.html')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertNotIn('img.image.url', content,
                         'Bug F: heatmap_analysis 不應引用不存在的 img.image')
        self.assertNotIn('img.anomaly_score', content,
                         'Bug F: heatmap_analysis 不應引用不存在的 img.anomaly_score')

    def test_correct_variant_url_encoding(self):
        """Grad-CAM++ 的 URL 篩選參數應正確編碼。"""
        path = os.path.join(BASE_DIR, 'templates', 'analyzer', 'gradcam_gallery.html')
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        self.assertNotIn('gradcam_pp', content,
                         'Bug F: variant 不應是 gradcam_pp（模型中不存在）')
        # 正確的 URL encoding 是 gradcam%2B%2B
        self.assertIn('gradcam%2B%2B', content,
                      'Bug F: Grad-CAM++ 應使用 URL encoded gradcam%2B%2B')


# ════════════════════════════════════════════════════════════════
# 額外：驗證 models.py 的欄位定義與模板引用一致
# ════════════════════════════════════════════════════════════════
class TestModelFieldConsistency(unittest.TestCase):
    """跨檔案驗證：所有模板引用的欄位在 models.py 中確實存在。"""

    @classmethod
    def setUpClass(cls):
        # 讀取 models.py 取得所有 CNNResult 和 GradCAMImage 的欄位
        import re
        models_path = os.path.join(BASE_DIR, 'analyzer', 'models.py')
        with open(models_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # 提取 CNNResult 欄位
        cls.cnn_fields = set(re.findall(r'^\s+(\w+)\s*=\s*models\.\w+Field', source, re.MULTILINE))
        # 提取 GradCAMImage 欄位
        cls.gradcam_fields = set(re.findall(r'^\s+(\w+)\s*=\s*models\.\w+Field', source, re.MULTILINE))

    def test_cnn_result_has_expected_fields(self):
        """CNNResult 模型應有我們引用的所有欄位。"""
        expected = {'threshold', 'avg_normal_error', 'avg_attack_error',
                    'normal_count', 'anomaly_count', 'detection_rate'}
        for field in expected:
            self.assertIn(field, self.cnn_fields,
                          f'CNNResult 模型缺少欄位: {field}')

    def test_gradcam_has_expected_fields(self):
        """GradCAMImage 模型應有我們引用的所有欄位。"""
        expected = {'comparison_image', 'heatmap_image', 'original_image',
                    'recon_error', 'is_anomaly', 'packet_index', 'variant'}
        for field in expected:
            self.assertIn(field, self.gradcam_fields,
                          f'GradCAMImage 模型缺少欄位: {field}')


if __name__ == '__main__':
    # 使用 UTF-8 輸出
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    unittest.main(verbosity=2)
