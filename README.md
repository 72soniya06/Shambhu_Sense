📘 Student Performance Predictor (Django App)
🎯 Overview

Student Performance Predictor is a Django web application that predicts a student’s final grade based on various academic and personal factors such as study hours, attendance, internal marks, and assignments completed.
It allows both students and teachers to interact with the model through a clean interface, upload CSV data for bulk predictions, and explore additional resources like syllabus and question papers.

🧠 Features
👩‍🎓 For Students

Manual entry form to predict performance.

Personalized study improvement tips based on prediction.

Upload a CSV file to predict grades for multiple students.

👨‍🏫 For Teachers

Same login system (no separate dashboard after login).

Option to train model (optional backend feature).

View predictions, manage uploaded data, and guide students.

🤖 Chatbot

Integrated chatbot to answer student-related queries and provide study help.

🔒 Authentication System

Username + password login.

Auto-create new users with temporary passwords.

Profile system (Student / Teacher roles supported).

📂 Database Models

Prediction: Stores all prediction records.

Profile: User details & role.

ChatHistory: Logs chatbot interactions.

