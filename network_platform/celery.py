# ============================================================
# network_platform/celery.py
# ============================================================
import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'network_platform.settings')

app = Celery('network_platform')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
