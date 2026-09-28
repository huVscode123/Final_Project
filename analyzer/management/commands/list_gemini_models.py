# -*- coding: utf-8 -*-
"""
analyzer/management/commands/list_gemini_models.py

用途：設定好 GEMINI_API_KEY 後，執行

    python manage.py list_gemini_models

列出這把 API Key 實際可用的 Gemini 模型名稱與支援的方法，用來確認
settings.GEMINI_MODEL 該填哪一個字串。Gemini 的可用模型會隨時間調整，
不同金鑰／地區可用的模型也可能不同，直接向 API 查詢會比查文件更準確。
"""
from django.core.management.base import BaseCommand, CommandError

from analyzer.ai_agent import gemini_client
from analyzer.ai_agent.gemini_client import GeminiConfigError


class Command(BaseCommand):
    help = "列出目前 GEMINI_API_KEY 可用的 Gemini 模型"

    def handle(self, *args, **options):
        try:
            models = gemini_client.list_models()
        except GeminiConfigError as e:
            raise CommandError(str(e))
        except Exception as e:
            raise CommandError(f"查詢失敗：{e}")

        if not models:
            self.stdout.write(self.style.WARNING("沒有查到任何可用模型，請確認 API Key 是否正確。"))
            return

        self.stdout.write(self.style.SUCCESS(f"共找到 {len(models)} 個可用模型：\n"))
        for m in models:
            name = m.get("name", "").replace("models/", "")
            methods = m.get("supportedGenerationMethods", [])
            marker = "✓" if "generateContent" in methods else " "
            self.stdout.write(f"  [{marker}] {name:<32} ({', '.join(methods)})")

        self.stdout.write(
            "\n提示：挑一個標示 [✓]（支援 generateContent）的模型名稱，填入 "
            "settings.py 的 GEMINI_MODEL，或設定環境變數 GEMINI_MODEL。"
        )
