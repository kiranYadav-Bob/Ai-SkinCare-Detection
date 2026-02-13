 🧠 AI SkinCare Analyst
🔬 Deep Learning Based Skin Disease Detection & AI Recommendation System
<p align="center"> <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"> <img src="https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow"> <img src="https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge&logo=streamlit"> <img src="https://img.shields.io/badge/Accuracy-81%25-brightgreen?style=for-the-badge"> <img src="https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge"> </p>
🚀 Project Overview

AI SkinCare Analyst is an end-to-end Deep Learning web application that detects multiple skin diseases from facial images and provides intelligent, personalized treatment recommendations.

The system integrates:

🧠 Custom CNN-based skin disease classifier

🖼️ Face detection & preprocessing pipeline

📊 Severity-based confidence scoring

🤖 LLM-powered recommendation engine

🌐 Interactive Streamlit deployment

This project demonstrates a real-world AI pipeline from raw image upload to actionable medical insights.

🏗️ System Architecture
<img width="850" height="323" alt="image" src="https://github.com/user-attachments/assets/fa9b4561-aa88-4519-9117-708f34983b4d" />
<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/1dd73df6-b43a-4509-ac30-da9cf96064aa" /><img width="1344" height="768" alt="Gemini_Generated_Image_zgiu8mzgiu8mzgiu" src="https://github.com/user-attachments/assets/4820ddc8-c7e3-4986-adb8-3a21f297f09c" />
## 🔄 End-to-End Workflow

> 👤 **User:** Uploads facial image  
>
> 🤖 **System:** Performs face detection using OpenCV  
>
> 🤖 **System:** Applies skin segmentation and preprocessing  
>
> 🧠 **CNN Model:** Predicts top skin conditions  
>
> 📊 **Severity Engine:** Calculates confidence threshold  
>
> 🤖 **AI Recommender:** Generates personalized skincare plan  
>
> ✅ **Output:** Displays diagnosis and treatment suggestions  


🧠 Model Details
## 🧠 Model Details

### 🔹 Architecture

- Conv2D (32 filters)
- Conv2D (64 filters)
- Conv2D (64 filters)
- MaxPooling Layers
- Flatten Layer
- Dense (64 neurons)
- Softmax Output Layer (8+ Skin Conditions)

---

### 🔹 Training Configuration

| Parameter | Value |
|------------|--------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 50 |
| Early Stopping | Enabled |
| Validation Accuracy | **78% – 81.7%** |
| Validation Loss | ~0.6 |


📊 Severity Classification (Chat Style)
## 📊 Severity Classification Logic

> 🔴 **High Confidence (>70%)**  
> Immediate dermatologist consultation recommended  
>
> 🟡 **Medium Confidence (50–70%)**  
> Targeted treatment and monitoring plan  
>
> 🟢 **Low Confidence (<50%)**  
> General skincare guidance and preventive care  

## 📂 Project Structure

```text
AI_based_skin_analyst/
│
├── ai_skincare_project/
│   ├── agentic_derm_flow.py
│   ├── app.py
│   ├── config.py
│   ├── face_crop.py
│   ├── image_upload.py
│   ├── image_utils.py
│   ├── llm_connector.py
│   ├── predict_skin_issue.py
│   └── prompt_templates.py
│
├── best_skin_model.h5
├── requirements.txt
├── README.md
├── .gitignore
└── .gitattributes
```

🛠️ Tech Stack (Clean List)
## 🛠️ Tech Stack

- 🐍 Python 3.10  
- 🧠 TensorFlow / Keras  
- 👁️ OpenCV  
- 🌐 Streamlit  
- 📊 NumPy / Pandas  
- 🔗 Git + Git LFS  



>
> 🖥️ How To Run
>## 🖥️ Run Locally

```bash
git clone https://github.com/kiranYadav-Bob/Ai-SkinCare-Detection.git
cd Ai-SkinCare-Detection
pip install -r requirements.txt
streamlit run ai_skincare_project/app.py

> 👨‍💻Author Section
## 👨‍💻 Author

**Kiran Kumar**  
MCA | AI & Machine Learning Enthusiast  
Passionate about building intelligent real-world AI solutions.


📌 Resume Summary Section
## 📌 Resume Summary

Developed an AI-powered multi-label skin disease detection system using TensorFlow CNN achieving 81% validation accuracy. Implemented severity scoring and LLM-assisted recommendation engine within an interactive Streamlit web application.

