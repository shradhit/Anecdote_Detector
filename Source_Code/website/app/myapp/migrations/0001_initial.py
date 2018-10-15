# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('email', models.CharField(default=b'', unique=b'True', max_length=60)),
                ('password', models.CharField(max_length=256)),
            ],
            options={
                'db_table': 'User',
            },
        ),
    ]
