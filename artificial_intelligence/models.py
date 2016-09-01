from __future__ import unicode_literals

from django.db import models

# Create your models here.

class Bactracking_Solution(models.Model):
    matrix_size = models.IntegerField(default=0)
    solution_cost = models.FloatField(default=0.0)
    coordinates = models.CharField(max_length=100000)
    expanded_nodes = models.IntegerField(default=0)
    execution_time = models.FloatField(default=0.0)
