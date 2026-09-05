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
    'unsupervised_vae': {'label': '非監督式 CNN-VAE', 'path': BASE_DIR / 'media' / 'model' / 'best_vae_model.pt', 'type': 'cnn_vae'},
    'semi_cicddos2019': {'label': '半監督式 CIC-DDoS2019', 'path': BASE_DIR / 'media' / 'model' / 'semi_cicddos2019.pt', 'type': 'hybrid_semi'},
    'semi_cicids2017': {'label': '半監督式 CICIDS2017', 'path': BASE_DIR / 'media' / 'model' / 'semi_cicids2017.pt', 'type': 'hybrid_semi'},
    'semi_nslkdd': {'label': '半監督式 NSL-KDD', 'path': BASE_DIR / 'media' / 'model' / 'semi_nslkdd.pt', 'type': 'hybrid_semi'},
}
# ── [P1-2 修正] 支援環境變數覆蓋 ──
CNN_THRESHOLD = float(os.getenv('CNN_THRESHOLD', '0.000069'))

DATA_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024
FILE_UPLOAD_MAX_MEMORY_SIZE = 500 * 1024 * 1024

REPORT_OUTPUT_DIR = BASE_DIR / 'media' / 'reports'
N8N_WEBHOOK_URL = os.getenv('N8N_WEBHOOK_URL', 'http://localhost:5678/webhook/ai-chat')
AI_CHAT_TIMEOUT = int(os.getenv('AI_CHAT_TIMEOUT', '30'))  # AI 代理人 webhook 逾時秒數

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
LOGIN_URL           = '/accounts/login/'
LOGIN_REDIRECT_URL  = '/analyzer/'
LOGOUT_REDIRECT_URL = '/accounts/login/'
