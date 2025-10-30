# predictor/models.py
# predictor/models.py
from django.contrib.auth.models import User
from django.db import models

# ------------------ Prediction Model ------------------
class Prediction(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    student_name = models.CharField(max_length=100)
    category = models.CharField(max_length=20)
    school_class = models.CharField(max_length=10, null=True, blank=True)
    course = models.CharField(max_length=50, null=True, blank=True)
    course_year = models.CharField(max_length=10, null=True, blank=True)
    percentage = models.FloatField(null=True, blank=True)
    cgpa = models.FloatField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    age = models.IntegerField(null=True, blank=True)
    study_hours = models.FloatField(null=True, blank=True)
    attendance = models.FloatField(null=True, blank=True)
    internal_marks = models.FloatField(null=True, blank=True)
    assignments_completed = models.IntegerField(null=True, blank=True)
    predicted_grade = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.student_name} ({self.category}) – {self.predicted_grade}"


# ------------------ Profile Model ------------------
# predictor/models.py
from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    ROLE_CHOICES = [
        ('student', 'Student'),
        ('teacher', 'Teacher'),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    roll_no = models.CharField(max_length=20, blank=True, null=True)
    phone = models.CharField(max_length=15, blank=True, null=True)
    school_or_college = models.CharField(max_length=100, blank=True, null=True)
    profile_pic = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    first_login = models.BooleanField(default=True)  # ✅ Flag for first login

    def __str__(self):
        return f"{self.user.username} - {self.role}"

from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        from .models import Profile
        Profile.objects.get_or_create(user=instance)

from django.db import models
from django.contrib.auth.models import User

class ChatHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    message = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.timestamp.strftime('%Y-%m-%d %H:%M')}"

class LibraryFile(models.Model):
    title = models.CharField(max_length=255)
    file_url = models.URLField()
    date_added = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.title


