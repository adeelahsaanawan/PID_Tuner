# Online PID Tuner App

**Live URL:** [https://pid-tuner.onrender.com/](https://pid-tuner.onrender.com/)

## Overview

The **Online PID Tuner App** is a free, web‑based tool for interactive PID controller analysis and tuning. Users can input plant parameters and PID controller gains to compute:

- **Bode Plot Data** (open-loop)
- **Step Response** (closed-loop)
- **Performance Metrics** such as gain margin, phase margin, rise time, settling time, overshoot, and steady‑state error

The application is built using Flask with the python‑control library on the backend and Bootstrap with Plotly on the frontend for a modern, interactive user experience.

## How It Works

1. **Input:**  
   Users enter the plant numerator/denominator coefficients and the PID parameters $K_p$, $K_i$, $K_d$, and the derivative filter coefficient $N$.

2. **Analysis:**  
   The backend creates the plant and the filtered PID controller, computes the Loop shape $L(s) = G(s) \cdot C(s)$ and forms the closed-loop transfer function  
   $T(s) = \frac{L(s)}{1 + L(s)}$.  
   It then simulates the step response and computes performance metrics along with extracting Bode plot data.

3. **Output:**  
   - An interactive Bode plot is rendered using Plotly.
   - The closed-loop step response is displayed interactively.
   - Performance metrics are shown in styled cards.

## Repository Structure
pid_tuner_app/
├── app.py           # Flask backend code
├── Procfile         # Render deployment configuration
├── requirements.txt # Python dependencies
└── templates/
    └── index.html   # Frontend (HTML/CSS/JS)

## Deployment

This app is hosted on **Render**. Simply push your changes to GitHub, and Render will automatically build and deploy your application using the provided `Procfile` and `requirements.txt`.

## Contact

Made by [Adeel Ahsan](mailto:maahsan@mun.ca)