from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'django-dev-key-請在正式環境替換成隨機字串')
DEBUG = os.getenv('DJANGO_DEBUG', 'true').lower() == 'true'
ALLOWED_HOSTS = os.getenv('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1,*').split(',')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'accounts',
    'projects',
    'analyzer',
    'api',
    'reports',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'network_platform.urls'

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],
    'APP_DIRS': True,
    'OPTIONS': {'context_processors': [
        'django.template.context_processors.debug',
        'django.template.context_processors.request',
        'django.contrib.auth.context_processors.auth',
        'django.contrib.messages.context_processors.messages',
    ]},
}]

WSGI_APPLICATION = 'network_platform.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'zh-hant'
TIME_ZONE     = 'Asia/Taipei'
USE_I18N = True
USE_TZ   = True

STATIC_URL       = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT      = BASE_DIR / 'staticfiles'
MEDIA_URL        = '/media/'
MEDIA_ROOT       = BASE_DIR / 'media'

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_RENDERER_CLASSES': [
        'rest_framework.renderers.JSONRenderer',
        'rest_framework.renderers.BrowsableAPIRenderer',
    ],
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

CORS_ALLOW_ALL_ORIGINS = DEBUG

CELERY_BROKER_URL         = os.getenv('CELERY_BROKER', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND     = os.getenv('CELERY_BACKEND', 'redis://localhost:6379/0')
CELERY_ACCEPT_CONTENT     = ['json']
CELERY_TASK_SERIALIZER    = 'json'
CELERY_RESULT_SERIALIZER  = 'json'
CELERY_TIMEZONE           = 'Asia/Taipei'
CELERY_TASK_ALWAYS_EAGER  = os.getenv('CELERY_EAGER', 'false').lower() == 'true'

CNN_MODEL_PATH  = BASE_DIR / 'media' / 'model' / 'best_model.pt'
CNN_LATENT_DIM  = 32
ANOMALY_MODELS = {
    # 每個選項指向獨立、已訓練的 checkpoint；不存在時 UI 會停用，不會改用其他模型。
    'unsup_cicids2017': {
        'label': '非監督式 CICIDS2017',
        'path': BASE_DIR / 'media' / 'model' / 'best_vae_cicids2017.pt',
        'type': 'cnn_vae', 'dataset': 'CICIDS2017', 'learning': 'unsupervised',
    },
    'semi_cicids2017': {
        'label': '半監督式 CICIDS2017',
        'path': BASE_DIR / 'media' / 'model' / 'semi_cicids2017.pt',
        'type': 'hybrid_semi', 'dataset': 'CICIDS2017', 'learning': 'semi_supervised',
    },
    'mta_malware': {
        'label': '非監督式 真實惡意軟體偵測 (MTA)',
        'path': BASE_DIR / 'media' / 'model' / 'best_vae_mta_malware.pt',
        'type': 'cnn_vae', 'dataset': 'MTA', 'learning': 'unsupervised',
    },
    'semi_mta_malware': {
        'label': '半監督式 真實惡意軟體偵測 (MTA)',
        'path': BASE_DIR / 'media' / 'model' / 'semi_mta_malware.pt',
        'type': 'hybrid_semi', 'dataset': 'MTA', 'learning': 'semi_supervised',
    },
    # NSL-KDD 保留舊紀錄相容性，但不提供給 PCAP 分析選擇。
    'semi_nslkdd': {
        'label': '半監督式 NSL-KDD（停用）',
        'path': BASE_DIR / 'media' / 'model' / 'semi_nslkdd.pt',
        'type': 'hybrid_semi', 'enabled': False, 'dataset': 'NSL-KDD',
    },
}

# Flow/CSV 模型必須沿用訓練期前處理器；packet-native checkpoint 不使用 scaler。
ANOMALY_PREPROCESSORS = {
    'unsup_cicids2017': (
        BASE_DIR / 'output' / 'model_cicids2017_vae_fixed'
        / 'cicids2017_preprocessor.pkl'
    ),
    'semi_cicids2017': (
        BASE_DIR / 'output' / 'model_semi_cicids2017'
        / 'cicids2017_preprocessor.pkl'
    ),
}

# ── [P1-2 修正] 支援環境變數覆蓋 ──
CNN_THRESHOLD = float(os.getenv('CNN_THRESHOLD', '0.000069'))

DATA_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024

REPORT_OUTPUT_DIR = BASE_DIR / 'media' / 'reports'
# ── AI Agent 設定（Gemini 直連，不需要 Docker / n8n）──────────
# 請在此填入您的 Google Gemini API Key，或透過環境變數設定
# 取得方式：https://aistudio.google.com/apikey
GEMINI_API_KEY          = os.getenv('GEMINI_API_KEY', 'your-api-key-here')  # ← 在此填入您的 API Key
GEMINI_MODEL            = os.getenv('GEMINI_MODEL', 'gemini-3.8-flash')
GEMINI_API_TIMEOUT      = int(os.getenv('GEMINI_API_TIMEOUT', '30'))
GEMINI_MAX_OUTPUT_TOKENS = int(os.getenv('GEMINI_MAX_OUTPUT_TOKENS', '1024'))
# 舊版相容欄位（部分程式碼仍可能讀取）
GEMINI_MODEL_NAME = GEMINI_MODEL

# 以下為舊版 n8n 設定（已棄用，保留向後相容）
N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL', 'http://localhost:5678/webhook/ai-chat')
AI_CHAT_TIMEOUT = int(os.getenv('AI_CHAT_TIMEOUT', '30'))

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
LOGIN_URL           = '/accounts/login/'
LOGIN_REDIRECT_URL  = '/analyzer/'
LOGOUT_REDIRECT_URL = '/accounts/login/'
