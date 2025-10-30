from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.conf import settings
import os, joblib, pandas as pd
from .forms import PredictForm, LoginForm, SetPasswordForm
from .models import Prediction
from .models import Profile
from .forms import LoginForm
import random, string


MODEL_FILE = os.path.join(settings.BASE_DIR, 'predictor', 'model_bundle.joblib')


def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)

def landing_page(request):
    return render(request, 'predictor/landing.html')



# ------------------ Role Selection ------------------
def role_select(request):
    return render(request, 'predictor/role_select.html')


# ------------------ Login View ------------------
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from .models import Profile
import random, string

# Helper function to generate random temporary password
def generate_temp_password(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username').strip()
        password = request.POST.get('password', '').strip()

        # First-time login: password left blank
        if password == "":
            # Check if user already exists
            user = User.objects.filter(username=username).first()
            if user:
                messages.info(request, "⚠ User already exists. Enter password to login.")
                return render(request, "predictor/login.html", {"username": username})
            else:
                # Create new user with temporary password
                temp_pass = generate_temp_password()
                user = User.objects.create_user(username=username, password=temp_pass)
                # Create profile safely
                profile, created = Profile.objects.get_or_create(user=user, defaults={'role':'student'})
                messages.success(request, f"✅ User created! Temporary password: {temp_pass}")
                return render(request, "predictor/login.html", {"username": username, "temp_password": temp_pass})

        # Normal login with password
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            # Redirect based on role
            if user.is_staff:
                return redirect('train')  # Teacher
            else:
                return redirect('predict_manual')  # Student
        else:
            messages.error(request, "❌ Invalid username or password")
            return render(request, "predictor/login.html", {"username": username})

    else:
        return render(request, "predictor/login.html")


    #------------------ Logout View ------------------
@login_required
def logout_view(request):
    logout(request)
    return redirect('login')


# ------------------ Context Helper for Sidebar ------------------
def get_user_context(request):
    """Returns context dictionary for sidebar/profile drawer"""
    return {
        'name': request.user.get_full_name() or request.user.username,
        'email': request.user.email,
        'phone': getattr(request.user, 'phone', ''),
        'school': getattr(request.user, 'school', ''),
        'category': 'Teacher' if request.user.is_staff else 'Student',
    }


# ------------------ Manual Prediction ------------------
@login_required
# ------------------ Manual Prediction ------------------
@login_required
def predict_manual(request):
    model_bundle = load_model()
    form = PredictForm(request.POST or None)
    result = None
    tips = None  # 🟢 add variable for study tips

    # --- Function to generate personalized tips ---
    def get_study_tips(grade):
        if grade >= 90:
            return "🌟 Excellent! Keep maintaining consistency and help others in studies to strengthen your knowledge."
        elif grade >= 75:
            return "💪 Great job! Try focusing more on weak subjects and regular revision to move towards excellence."
        elif grade >= 60:
            return "📘 Good effort! Increase your daily study hours slightly and revise weekly to improve your score."
        elif grade >= 45:
            return "📈 You’re improving! Focus on completing assignments on time and maintaining attendance."
        else:
            return "⚠ Needs improvement! Stay consistent, attend all classes, and ask for help when needed."

    if request.method == 'POST' and form.is_valid():
        if not model_bundle:
            messages.error(request, "⚠ Model not trained yet. Please train it first.")
        else:
            pipeline = model_bundle['pipeline']
            features = model_bundle['features']

            # Get category (school or college)
            category = form.cleaned_data.get('category')

            # Map common fields
            data = {
                'gender': form.cleaned_data['gender'],
                'age': form.cleaned_data['age'],
                'study_hours': form.cleaned_data['study_hours'] or 0,
                'attendance': form.cleaned_data['attendance'],
                'internal_marks': form.cleaned_data['internal_marks'],
                'assignments_completed': form.cleaned_data['assignments_completed'],
            }

            # 🟢 Add category-specific numeric values
            if category == 'school':
                data['percentage'] = form.cleaned_data.get('percentage') or 0
                data['cgpa'] = 0
            else:
                data['cgpa'] = form.cleaned_data.get('cgpa') or 0
                data['percentage'] = 0

            # Convert to DataFrame
            df = pd.DataFrame([data])

            # Ensure feature alignment
            for col in features:
                if col not in df.columns:
                    df[col] = 0
            df = df[features]

            # 🧮 Make prediction
            pred = pipeline.predict(df)[0]
            result = round(float(pred), 2)

            # 🎯 Generate study tips based on performance
            tips = get_study_tips(result)

            # 💾 Save prediction in DB
            Prediction.objects.create(
                student_name=form.cleaned_data['student_name'],
                gender=form.cleaned_data['gender'],
                age=form.cleaned_data['age'],
                study_hours=form.cleaned_data['study_hours'] or 0,
                attendance=form.cleaned_data['attendance'],
                internal_marks=form.cleaned_data['internal_marks'],
                assignments_completed=form.cleaned_data['assignments_completed'],
                predicted_grade=result,
            )

    context = get_user_context(request)
    context.update({
        'form': form,
        'result': result,
        'tips': tips,  # 🟢 add to context
    })
    return render(request, 'predictor/predict_manual.html', context)


# ------------------ CSV Upload Prediction ------------------
@login_required
# ------------------ CSV Upload Prediction (Updated for manual fields) ------------------
@login_required
def predict_csv(request):
    predicted_file_url = None

    # 🧠 Step 1: Check if model exists
    if not os.path.exists(MODEL_FILE):
        messages.error(request, "⚠ Model not trained yet. Please train it first.")
        context = get_user_context(request)
        context.update({'predicted_file_url': predicted_file_url})
        return render(request, 'predictor/predict_csv.html', context)

    # Load model
    model_bundle = load_model()
    pipeline = model_bundle['pipeline']
    features = model_bundle['features']

    # 🧩 Step 2: Handle uploaded CSV
    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

        input_path = os.path.join(settings.MEDIA_ROOT, csv_file.name)
        with open(input_path, 'wb+') as destination:
            for chunk in csv_file.chunks():
                destination.write(chunk)

        try:
            # 🧾 Read uploaded CSV
            df = pd.read_csv(input_path)
            df.columns = df.columns.str.strip().str.lower()

            expected_cols = [
                'student name', 'category', 'class', 'gender', 'age',
                'study hours', 'attendance', 'internal marks',
                'assignments completed', 'percentage', 'cgpa'
            ]

            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                messages.error(request, f"⚠ Missing columns in CSV: {', '.join(missing)}")
            else:
                # 🧠 Prepare model input
                rows = []
                for _, row in df.iterrows():
                    data = {
                        'gender': row.get('gender', ''),
                        'age': row.get('age', 0),
                        'study_hours': row.get('study hours', 0) or 0,
                        'attendance': row.get('attendance', 0),
                        'internal_marks': row.get('internal marks', 0),
                        'assignments_completed': row.get('assignments completed', 0),
                    }

                    # Category logic
                    category = str(row.get('category', '')).lower()
                    if category == 'school':
                        data['percentage'] = row.get('percentage', 0)
                        data['cgpa'] = 0
                    else:
                        data['cgpa'] = row.get('cgpa', 0)
                        data['percentage'] = 0

                    rows.append(data)

                df_features = pd.DataFrame(rows)

                # Align with model features
                for col in features:
                    if col not in df_features.columns:
                        df_features[col] = 0
                df_features = df_features[features]

                # 🔮 Predict
                predictions = pipeline.predict(df_features)
                df['Predicted_Final_Grade'] = [round(float(p), 2) for p in predictions]

                # Save output
                output_filename = f"predicted_{csv_file.name}"
                output_path = os.path.join(settings.MEDIA_ROOT, output_filename)
                df.to_csv(output_path, index=False)
                predicted_file_url = settings.MEDIA_URL + output_filename

                messages.success(request, "✅ Prediction completed successfully!")

        except Exception as e:
            messages.error(request, f"❌ Error processing CSV: {str(e)}")

    # Render
    context = get_user_context(request)
    context.update({'predicted_file_url': predicted_file_url})
    return render(request, 'predictor/predict_csv.html', context)



# ------------------ Teacher: Train model ------------------
@login_required
def train_view(request):
    msg = None
    if request.method == "POST":
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline

            data_path = os.path.join(settings.BASE_DIR, 'student_data.csv')
            if not os.path.exists(data_path):
                msg = "❌ Dataset not found in 'student_data.csv'"
            else:
                data = pd.read_csv(data_path)
                features = ['gender', 'age', 'study_hours', 'attendance', 'internal_marks', 'assignments_completed']
                target = 'final_grade'

                X = pd.get_dummies(data[features], drop_first=True)
                y = data[target]

                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', LinearRegression())
                ])
                pipe.fit(X, y)

                model_bundle = {'pipeline': pipe, 'features': X.columns.tolist()}
                os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
                joblib.dump(model_bundle, MODEL_FILE)

                msg = "✅ Model trained & saved successfully!"
        except Exception as e:
            msg = f"❌ Error: {str(e)}"

    context = get_user_context(request)
    context.update({'msg': msg})
    return render(request, 'predictor/train.html', context)


# ------------------ Download File ------------------
from django.http import FileResponse, Http404

@login_required
def download_file(request, filename):
    """
    Allows users to download predicted CSV files securely.
    """
    file_path = os.path.join(settings.MEDIA_ROOT, filename)
    if os.path.exists(file_path):
        return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=filename)
    else:
        raise Http404("File not found")


# ------------------ Set Password View ------------------
@login_required
def set_password_view(request):
    user_id = request.session.get('temp_user_id')
    user = User.objects.filter(id=user_id).first() if user_id else None

    if not user:
        messages.error(request, "❌ Invalid access. Please log in first.")
        return redirect('login')

    form = SetPasswordForm(request.POST or None)

    if request.method == "POST" and form.is_valid():
        password = form.cleaned_data['password1']
        user.set_password(password)
        user.save()

        # auto-create profile
        Profile.objects.get_or_create(user=user)

        del request.session['temp_user_id']  # clear session
        messages.success(request, "✅ Password created successfully! You can now log in.")
        return redirect('login')

    return render(request, 'predictor/set_password.html', {'form': form, 'user': user})
# ------------------ Profile Views ------------------
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from .models import Profile
from .forms import ProfileForm

@login_required
def profile_view(request):
    profile, created = Profile.objects.get_or_create(user=request.user)
    return render(request, 'predictor/profile.html', {'profile': profile})

@login_required
def edit_profile(request):
    profile, created = Profile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        form = ProfileForm(request.POST, request.FILES, instance=profile, user=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "✅ Profile updated successfully")
            return redirect('profile_view')
        else:
            messages.error(request, "❌ Please correct the errors below")
    else:
        form = ProfileForm(instance=profile, user=request.user)
    return render(request, 'predictor/edit_profile.html', {'form': form})

# views.py
from .models import Profile


# ------------------ Context Helper for Sidebar ------------------
def get_user_context(request):
    """Returns context dictionary for sidebar/profile drawer"""
    profile = None
    if request.user.is_authenticated:
        profile, created = Profile.objects.get_or_create(user=request.user)

    return {
        'name': request.user.get_full_name() or request.user.username if request.user.is_authenticated else "Guest",
        'email': request.user.email if request.user.is_authenticated else "",
        'phone': profile.phone if profile else "",
        'school': profile.school_or_college if profile else "",
        'role': profile.role if profile else "",
        'profile_pic': profile.profile_pic.url if profile and profile.profile_pic else None,
    }

from django.contrib.auth.models import User

def register_user(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")

        # ✅ Create user without password first
        user = User.objects.create(username=username, email=email)
        user.set_unusable_password()   # 👈 Important line
        user.save()

        messages.success(request, "✅ User created! Please set password on first login.")
        return redirect("login")

    return render(request, "predictor/register.html")

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from .models import ChatHistory
import requests, os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# ✅ Chatbot main page
@login_required
def chatbot_page(request):
    return render(request, "predictor/chatbot.html")

# ✅ Handle AI responses
@csrf_exempt
@login_required
def chatbot_api(request):
    if request.method == "POST":
        user_message = request.POST.get("message", "").strip()
        if not user_message:
            return JsonResponse({"error": "No message provided."})

        try:
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"role": "user", "parts": [{"text": user_message}]}]
            }

            response = requests.post(url, headers=headers, json=payload)
            result = response.json()

            if "candidates" in result:
                ai_reply = result["candidates"][0]["content"]["parts"][0]["text"]

                # ✅ Save in ChatHistory
                ChatHistory.objects.create(
                    user=request.user,
                    message=user_message,
                    response=ai_reply
                )

                return JsonResponse({"answer": ai_reply})

            elif "error" in result:
                return JsonResponse({"error": result["error"].get("message", "API error.")})

            return JsonResponse({"error": "Unexpected response format."})

        except Exception as e:
            return JsonResponse({"error": str(e)})

    return JsonResponse({"error": "Invalid request method."})

# ✅ Show chat history
@login_required
def chat_history(request):
    chats = ChatHistory.objects.filter(user=request.user).order_by("-timestamp")
    return render(request, "predictor/chat_history.html", {"chats": chats})

# ✅ Delete single chat
@login_required
def delete_chat(request, chat_id):
    ChatHistory.objects.filter(id=chat_id, user=request.user).delete()
    return redirect("chat_history")

import os
print("Gemini key loaded:", os.getenv("GEMINI_API_KEY"))



# predictor/views.py

from django.shortcuts import render

def chatbot_page(request):
    return render(request, 'predictor/chatbot.html')

from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect
from django.contrib import messages
from .models import LibraryFile

@login_required
def add_to_library(request, title, file_url):
    LibraryFile.objects.create(
        user=request.user,
        title=title,
        file_url=file_url
    )
    messages.success(request, f"📚 '{title}' saved to your library!")
    return redirect('my_library')

@login_required
def my_library(request):
    files = LibraryFile.objects.filter(user=request.user).order_by('-uploaded_at')
    return render(request, 'predictor/my_library.html', {'files': files})

@login_required
def delete_from_library(request, file_id):
    file = LibraryFile.objects.get(id=file_id, user=request.user)
    file.delete()
    messages.success(request, "🗑️ File removed from your library.")
    return redirect('my_library')

from django.shortcuts import render, redirect
from .models import LibraryFile
from django.contrib.auth.decorators import login_required

from django.shortcuts import render

def search_papers(request):
    query = request.GET.get('query', '')
    results = []

    # Example: you can later connect this with database or external API
    if query:
        # Just demo results (replace with actual logic)
        results = [
            {"title": f"{query} - PYQ 2024", "link": "#"},
            {"title": f"{query} - Notes", "link": "#"},
            {"title": f"{query} - Assignment", "link": "#"},
        ]

    return render(request, 'predictor/search_results.html', {'query': query, 'results': results})


