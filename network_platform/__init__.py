# ============================================================
# network_platform/__init__.py
# 安全載入 Celery — 若 celery 未安裝，則以同步模式運行
# ============================================================
try:
    from .celery import app as celery_app
    __all__ = ('celery_app',)
except ImportError:
    celery_app = None
    __all__ = ()
