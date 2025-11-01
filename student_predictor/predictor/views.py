# predictor/views.py

import os, joblib, pandas as pd, random, string, requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.db.models import Q
from django.http import FileResponse, Http404, JsonResponse
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv

from .forms import PredictForm, LoginForm, SetPasswordForm, ProfileForm
from .models import Prediction, Profile, ChatHistory

load_dotenv()
MODEL_FILE = os.path.join(settings.BASE_DIR, 'predictor', 'model_bundle.joblib')
api_key = os.getenv("GEMINI_API_KEY")


# ----------------------------------------------------
# 🔹 Utility functions
# ----------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_FILE):
        return None
    return joblib.load(MODEL_FILE)


def generate_temp_password(length=8):
    return ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(length))


def get_user_context(request):
    """Sidebar & profile drawer data"""
    profile, _ = Profile.objects.get_or_create(user=request.user)
    return {
        'name': request.user.get_full_name() or request.user.username,
        'email': request.user.email,
        'phone': profile.phone,
        'school': profile.school_or_college,
        'role': profile.role,
        'profile_pic': profile.profile_pic.url if profile.profile_pic else None,
    }


# ----------------------------------------------------
# 🔹 Basic pages
# ----------------------------------------------------
def landing_page(request):
    return render(request, 'predictor/landing.html')


def role_select(request):
    return render(request, 'predictor/role_select.html')


# ----------------------------------------------------
# 🔹 Authentication
def login_view(request):
    if request.method == "POST":
        username = request.POST.get('username').strip()
        password = request.POST.get('password', '').strip()

        if password == "":
            user = User.objects.filter(username=username).first()
            if user:
                messages.info(request, "⚠ User already exists. Enter password to login.")
                return render(request, "predictor/login.html", {"username": username})
            else:
                temp_pass = generate_temp_password()
                user = User.objects.create_user(username=username, password=temp_pass)
                Profile.objects.get_or_create(user=user, defaults={'role': 'student'})
                messages.success(request, f"✅ User created! Temporary password: {temp_pass}")
                return render(request, "predictor/login.html", {"username": username, "temp_password": temp_pass})

        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            # 🔹 Redirect all users (teacher or student) to manual prediction page
            return redirect('predict_manual')
        else:
            messages.error(request, "❌ Invalid username or password")

    return render(request, "predictor/login.html")



@login_required
def logout_view(request):
    logout(request)
    return redirect('login')


# ----------------------------------------------------
# 🔹 Manual Prediction with tips
# ----------------------------------------------------
@login_required
def predict_manual(request):
    model_bundle = load_model()
    form = PredictForm(request.POST or None)
    result, tips = None, None

    def get_study_tips(grade):
        if grade >= 90:
            return "🌟 Excellent! Keep it up and help others to strengthen your concepts."
        elif grade >= 75:
            return "💪 Great! Focus more on weak topics and maintain consistency."
        elif grade >= 60:
            return "📘 Good effort! Increase daily study time and do regular revisions."
        elif grade >= 45:
            return "📈 You’re improving! Stay punctual and finish all assignments."
        else:
            return "⚠ Needs improvement! Attend all classes and ask teachers for help."

    if request.method == 'POST' and form.is_valid():
        if not model_bundle:
            messages.error(request, "⚠ Model not trained yet.")
        else:
            pipeline = model_bundle['pipeline']
            features = model_bundle['features']

            data = {
                'gender': form.cleaned_data['gender'],
                'age': form.cleaned_data['age'],
                'study_hours': form.cleaned_data['study_hours'] or 0,
                'attendance': form.cleaned_data['attendance'],
                'internal_marks': form.cleaned_data['internal_marks'],
                'assignments_completed': form.cleaned_data['assignments_completed'],
            }

            if form.cleaned_data['category'] == 'school':
                data['percentage'] = form.cleaned_data.get('percentage') or 0
                data['cgpa'] = 0
            else:
                data['cgpa'] = form.cleaned_data.get('cgpa') or 0
                data['percentage'] = 0

            df = pd.DataFrame([data])
            for col in features:
                if col not in df.columns:
                    df[col] = 0
            df = df[features]

            pred = pipeline.predict(df)[0]
            result = round(float(pred), 2)
            tips = get_study_tips(result)

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
    context.update({'form': form, 'result': result, 'tips': tips})
    return render(request, 'predictor/predict_manual.html', context)


# ----------------------------------------------------
# 🔹 CSV Upload Prediction
# ----------------------------------------------------
@login_required
def predict_csv(request):
    predicted_file_url = None
    if not os.path.exists(MODEL_FILE):
        messages.error(request, "⚠ Model not trained yet.")
        return render(request, 'predictor/predict_csv.html', get_user_context(request))

    model_bundle = load_model()
    pipeline, features = model_bundle['pipeline'], model_bundle['features']

    if request.method == 'POST' and 'csv_file' in request.FILES:
        csv_file = request.FILES['csv_file']
        path = os.path.join(settings.MEDIA_ROOT, csv_file.name)
        with open(path, 'wb+') as dest:
            for chunk in csv_file.chunks():
                dest.write(chunk)

        try:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip().str.lower()
            if 'final grade' not in df.columns:
                df['final grade'] = 0

            rows = []
            for _, row in df.iterrows():
                data = {
                    'gender': row.get('gender', ''),
                    'age': row.get('age', 0),
                    'study_hours': row.get('study hours', 0),
                    'attendance': row.get('attendance', 0),
                    'internal_marks': row.get('internal marks', 0),
                    'assignments_completed': row.get('assignments completed', 0),
                    'percentage': row.get('percentage', 0),
                    'cgpa': row.get('cgpa', 0),
                }
                rows.append(data)

            df_features = pd.DataFrame(rows)
            for col in features:
                if col not in df_features.columns:
                    df_features[col] = 0
            df_features = df_features[features]

            df['Predicted_Final_Grade'] = [round(float(p), 2) for p in pipeline.predict(df_features)]
            output = f"predicted_{csv_file.name}"
            df.to_csv(os.path.join(settings.MEDIA_ROOT, output), index=False)
            predicted_file_url = settings.MEDIA_URL + output
            messages.success(request, "✅ Prediction completed!")

        except Exception as e:
            messages.error(request, f"❌ Error: {e}")

    context = get_user_context(request)
    context['predicted_file_url'] = predicted_file_url
    return render(request, 'predictor/predict_csv.html', context)


# ----------------------------------------------------
# 🔹 Teacher: Train Model
# ----------------------------------------------------
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
                msg = "❌ Dataset not found."
            else:
                df = pd.read_csv(data_path)
                features = ['gender', 'age', 'study_hours', 'attendance', 'internal_marks', 'assignments_completed']
                X = pd.get_dummies(df[features], drop_first=True)
                y = df['final_grade']

                pipe = Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())])
                pipe.fit(X, y)
                joblib.dump({'pipeline': pipe, 'features': X.columns.tolist()}, MODEL_FILE)
                msg = "✅ Model trained & saved!"
        except Exception as e:
            msg = f"❌ {e}"

    ctx = get_user_context(request)
    ctx['msg'] = msg
    return render(request, 'predictor/train.html', ctx)


# ----------------------------------------------------
# 🔹 File Download + Library System
# ----------------------------------------------------
#@login_required
#def download_file(request, filename):
    path = os.path.join(settings.MEDIA_ROOT, filename)
    if os.path.exists(path):
        # Auto-save to library
        LibraryFile.objects.get_or_create(
            user=request.user,
            title=filename,
            file_url=settings.MEDIA_URL + filename
        )
        return FileResponse(open(path, 'rb'), as_attachment=True, filename=filename)
    raise Http404("File not found")



# ----------------------------------------------------
# 🔹 Chatbot (Gemini)
# ----------------------------------------------------
@login_required
def chatbot_page(request):
    return render(request, "predictor/chatbot.html")


@csrf_exempt
@login_required
def chatbot_api(request):
    if request.method == "POST":
        user_msg = request.POST.get("message", "").strip()
        if not user_msg:
            return JsonResponse({"error": "No message provided."})

        try:
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent?key={api_key}"
            payload = {"contents": [{"role": "user", "parts": [{"text": user_msg}]}]}
            res = requests.post(url, headers={"Content-Type": "application/json"}, json=payload)
            data = res.json()

            if "candidates" in data:
                reply = data["candidates"][0]["content"]["parts"][0]["text"]
                ChatHistory.objects.create(user=request.user, message=user_msg, response=reply)
                return JsonResponse({"answer": reply})
            return JsonResponse({"error": "Unexpected response."})
        except Exception as e:
            return JsonResponse({"error": str(e)})

    return JsonResponse({"error": "Invalid method"})


@login_required
def chat_history(request):
    chats = ChatHistory.objects.filter(user=request.user).order_by("-timestamp")
    return render(request, "predictor/chat_history.html", {"chats": chats})


@login_required
def delete_chat(request, chat_id):
    ChatHistory.objects.filter(id=chat_id, user=request.user).delete()
    return redirect("chat_history")

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

