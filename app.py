from flask import Flask, render_template, request, jsonify
import numpy as np
import io, base64
import matplotlib.pyplot as plt
import traceback
import control
from control import tf, margin, step_response, feedback, dcgain
from control.timeresp import step_info

app = Flask(__name__)

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
        # Combined to a single rational function:
        #   Numerator: [N*(Kp+Kd), (Kp + Ki*N), Ki]
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

        # Simulate step response (0 to 10 sec)
        t = np.linspace(0, 10, 1000)
        t_out, y_out = step_response(T, T=t)

        # Compute performance metrics using step_info (returns a dictionary)
        info = step_info(T)
        ss_val = dcgain(T)
        steady_state_error = abs(1 - ss_val)

        result = {
            "gain_margin_dB": gain_margin_dB,
            "phase_margin_deg": pm,
            "wcg": wcg,
            "wcp": wcp,
            "rise_time": info["RiseTime"],
            "settling_time": info["SettlingTime"],
            "overshoot": info["Overshoot"],
            "steady_state_error": steady_state_error,
            "step_response": {
                "time": t_out.tolist(),
                "response": y_out.tolist()
            }
        }

        # Generate a Bode plot image as PNG (encoded in base64)
        fig, ax = plt.subplots(2, 1, figsize=(6,8))
        # Call the standalone bode() function with the correct parameter name: plot=False
        mag, phase, omega = control.bode(L, dB=True, plot=False)
        ax[0].semilogx(omega, 20 * np.log10(mag))
        ax[0].set_title("Magnitude (dB)")
        ax[1].semilogx(omega, phase * 180/np.pi)
        ax[1].set_title("Phase (deg)")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        bode_image = base64.b64encode(buf.read()).decode("utf-8")
        result["bode_plot"] = bode_image

        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
