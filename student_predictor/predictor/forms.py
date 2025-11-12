# predictor/forms.py
# predictor/forms.py
from django import forms

class SetPasswordForm(forms.Form):
    password1 = forms.CharField(widget=forms.PasswordInput, label="New Password")
    password2 = forms.CharField(widget=forms.PasswordInput, label="Confirm Password")


    def clean(self):
        cleaned_data = super().clean()
        p1 = cleaned_data.get("password1")
        p2 = cleaned_data.get("password2")
        if p1 != p2:
            raise forms.ValidationError("Passwords do not match")
        return cleaned_data


GENDER_CHOICES = [
    ('M', 'Male'),
    ('F', 'Female'),
    ('O', 'Other'),
]

CATEGORY_CHOICES = [
    ('school', 'School'),
    ('college', 'College'),
]

CLASS_CHOICES = [(str(i), f'Class {i}') for i in range(1, 13)]

COURSE_CHOICES = [
    ('btech', 'B.Tech (4 Years)'),
    ('mba', 'MBA (2 Years)'),
    ('bca', 'BCA (3 Years)'),
    ('bpharm', 'B.Pharm (4 Years)'),
    ('dpharm', 'D.Pharm (2 Years)'),
]

YEAR_CHOICES = [(str(i), f'Year {i}') for i in range(1, 5)]


class PredictForm(forms.Form):
    student_name = forms.CharField(max_length=100, label="Student Name")
    category = forms.ChoiceField(choices=CATEGORY_CHOICES, label="Category")

    # For school students
    school_class = forms.ChoiceField(choices=CLASS_CHOICES, required=False, label="Class (1–12)")

    # For college students
    course = forms.ChoiceField(choices=COURSE_CHOICES, required=False, label="Course")
    course_year = forms.ChoiceField(choices=YEAR_CHOICES, required=False, label="Year")

    # Common fields
    gender = forms.ChoiceField(choices=GENDER_CHOICES)
    age = forms.IntegerField(min_value=5, max_value=100)

    # ✅ study_hours optional
    study_hours = forms.FloatField(min_value=0, required=False, label="Study Hours (optional)")

    attendance = forms.FloatField(min_value=0, max_value=100)

    # ✅ replaced previous_grade → internal_marks
    internal_marks = forms.FloatField(min_value=0, max_value=100, label="Internal Marks")

    assignments_completed = forms.IntegerField(min_value=0, max_value=10, label="Assignment_Completed")

    # Category-specific fields (school: percentage, college: cgpa)
    percentage = forms.FloatField(min_value=0, max_value=100, required=False, label="Percentage (for school)")
    cgpa = forms.FloatField(min_value=0, max_value=10, required=False, label="CGPA (for college)")

from django import forms

class LoginForm(forms.Form):
    username = forms.CharField(max_length=100)
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'required': False}),  # allow blank for first-time login
        required=False
    )


# Add this at the bottom of predictor/forms.py

from django import forms
from .models import Profile
from django.contrib.auth.models import User

class ProfileForm(forms.ModelForm):
    # User model fields
    first_name = forms.CharField(max_length=30, required=True, label="First Name")
    last_name = forms.CharField(max_length=30, required=True, label="Last Name")
    email = forms.EmailField(max_length=254, required=True, label="Email Address")

    class Meta:
        model = Profile
        fields = ['phone', 'roll_no', 'school_or_college', 'role', 'profile_pic']

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)
        if self.user:
            self.fields['first_name'].initial = self.user.first_name
            self.fields['last_name'].initial = self.user.last_name
            self.fields['email'].initial = self.user.email

    def save(self, commit=True):
        profile = super().save(commit=False)
        if self.user:
            # Save User fields
            self.user.first_name = self.cleaned_data['first_name']
            self.user.last_name = self.cleaned_data['last_name']
            self.user.email = self.cleaned_data['email']
            if commit:
                self.user.save()
                profile.user = self.user
                profile.save()
        return profile



