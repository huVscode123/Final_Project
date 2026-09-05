from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [('analyzer', '0001_initial')]
    operations = [migrations.AddField(model_name='analysissession', name='model_key', field=models.CharField(default='unsupervised_vae', max_length=30, verbose_name='異常偵測模型'))]
