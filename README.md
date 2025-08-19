# ML Model Inference with FastAPI

## 📌 Project Description
This project demonstrates how to deploy a Machine Learning model using **FastAPI** for real-time inference. Users can send input data to the API and get predictions instantly.

## 🚀 Features
- Fast and lightweight API
- Easy integration with trained ML models
- Interactive Swagger UI available at `/docs` endpoint
- Example requests included for testing

## 🛠️ Technologies Used
- Python
- FastAPI
- Uvicorn
- Scikit-learn / Pandas (for ML model)
- Git & GitHub (for version control)

## ⚙️ How to Run Locally
1. Clone this repository:  
   ```bash
   git clone https://github.com/kawyasathsarani/ML-Model-Inference-with-FastAPI.git

2.Navigate to the project folder:
    
    cd ML-Model-Inference-with-FastAPI

3.Install dependencies:
    
    pip install -r requirements.txt

4.Run the FastAPI app:
    
    uvicorn main:app --reload

5.Open your browser and go to:
    
    http://127.0.0.1:8000/docs

📂 Project Structure

ML-Model-Inference-with-FastAPI/
│
├─ main.py             # FastAPI app
├─ model.pkl           # Trained ML model
├─ requirements.txt    # Python dependencies
├─ README.md           # Project description


💡Send a POST request to /predict with JSON data like:

{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}


👩‍💻 Author
Kawya Sathsarani


📄 License

### **To update your README on GitHub:**
1. Save this text as `README.md` in your project folder.  
2. Run in terminal:  
```bash
git add README.md
git commit -m "Update README with license link and full details"
git push
