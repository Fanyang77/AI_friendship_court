# 🦉 Friendship Court – LLM Conflict Mediator

Friendship Court is a small Streamlit web app where a very patient owl judge helps you look at a conflict more clearly.

You paste two perspectives on the same situation (Person A and Person B), and the app uses an LLM to:

- write a **neutral summary** of what happened  
- split **responsibility** between Person A and Person B (in %)  
- give **concrete advice** for each person  
- generate a reusable **apology template**  
- optionally raise a **safety note** if the situation involves abuse, self-harm, or other serious issues  

The UI includes:

- a **multi-step flow** (input → owl thinking → results)  
- a full-screen **“owl is thinking”** animation while the model runs  
- a **clean card layout** for summary & advice  
- an **Altair bar chart** to visualize responsibility split

---

## ✨ Features

- 🧠 **LLM-powered conflict mediation** (OpenAI Chat Completions API)  
- 🧾 **Neutral recap** of the situation  
- 📊 **Responsibility breakdown** for Person A & Person B  
- 💡 **Practical advice** for each side  
- 📝 **Apology script** you can customize  
- 🚨 **Safety flag** & message for serious issues  
- 🎨 Custom background image and cute owl illustration  
- 🎬 Optional full-screen “thinking” animation while the LLM runs

If the LLM call fails for any reason (network, auth, bad JSON), the app falls back to a simple **mock heuristic** that splits responsibility based on story length and returns generic advice.

---

## 🛠 Tech Stack

- **Python**
- **Streamlit** – web UI
- **Altair** – responsibility bar chart
- **Pandas** – chart data prep
- **OpenAI Python SDK** – LLM calls
- **python-dotenv** – load `OPENAI_API_KEY` from `.env`
- Standard library: `json`, `dataclasses`, `typing`, `pathlib`, `time`, `base64`

---

## 📁 Project Structure

Typical layout:

```text
project-root/
├─ app.py                    # this code
├─ requirements.txt
├─ README.md
└─ assets/
   ├─ app_background.jpg     # background image for the app
   ├─ owl_judge.png          # cute owl judge illustration
   └─ loading.mov            # “owl is thinking” animation (mp4/mov)
