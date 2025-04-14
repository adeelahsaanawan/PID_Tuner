from flask import Flask, render_template, request, jsonify
import numpy as np
import io, base64
import matplotlib.pyplot as plt
import traceback
import control
from control import tf, margin, step_response, feedback, dcgain
from control.timeresp import step_info

app = Flask(__name__)

def sanitize(val):
    """Convert non-finite numbers (NaN/Infinity) to None."""
    if np.isscalar(val) and not np.isfinite(val):
        return None
    return val

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Get data from the request JSON
        data = request.get_json()
        plant_num = [float(x) for x in data.get("plant_num", "").split(",")]
        plant_den = [float(x) for x in data.get("plant_den", "").split(",")]
        kp = float(data.get("kp", 1))
        ki = float(data.get("ki", 0.5))
        kd = float(data.get("kd", 0.1))
        N = float(data.get("N", 10))  # Derivative filter coefficient

        # Build the plant G(s)
        G = tf(plant_num, plant_den)

        # Build the filtered PID controller:
        # C(s) = Kp + Ki/s + (Kd * N * s)/(1+N * s)
        # Combined as a single rational function:
        #   Numerator: [N*(Kp+Kd), (Kp+Ki*N), Ki]
        #   Denom: [N, 1, 0]
        C_num = [N * (kp + kd), (kp + ki * N), ki]
        C_den = [N, 1, 0]
        C = tf(C_num, C_den)

        # Open-loop transfer function L(s) = G(s)*C(s)
        L = G * C

        # Closed-loop transfer function T(s) = L(s) / [1 + L(s)]
        T = feedback(L, 1)

        # Compute gain and phase margins.
        gm, pm, wcg, wcp = margin(L)
        gain_margin_dB = 20 * np.log10(gm) if gm > 0 else None

        # Sanitize outputs to remove non-finite values
        gain_margin_dB = sanitize(gain_margin_dB)
        pm = sanitize(pm)
        wcg = sanitize(wcg)
        wcp = sanitize(wcp)

        # Simulate step response (0 to 10 sec)
        t = np.linspace(0, 10, 1000)
        t_out, y_out = step_response(T, T=t)

        # Compute performance metrics using step_info (returns a dictionary)
        info = step_info(T)
        ss_val = dcgain(T)
        steady_state_error = abs(1 - ss_val)

        result = {
            "gain_margin_dB": gain_margin_dB,
            "phase_margin_deg": sanitize(pm),
            "wcg": sanitize(wcg),
            "wcp": sanitize(wcp),
            "rise_time": sanitize(info["RiseTime"]),
            "settling_time": sanitize(info["SettlingTime"]),
            "overshoot": sanitize(info["Overshoot"]),
            "steady_state_error": sanitize(steady_state_error),
            "step_response": {
                "time": t_out.tolist(),
                "response": y_out.tolist()
            }
        }

        # Generate Bode plot data using control.bode() (earlier method)
        # Use the bode() function with plot=False to get the arrays.
        # With dB=True, the magnitude is already in dB.
        mag, phase, omega = control.bode(L, dB=True, plot=False)
        # Convert phase from radians to degrees.
        phase_deg = (phase * 180 / np.pi).tolist()
        bode_data = {
            "omega": omega.tolist(),
            "magnitude_db": mag.tolist(),
            "phase_deg": phase_deg
        }
        result["bode_data"] = bode_data

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
