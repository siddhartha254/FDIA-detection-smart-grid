# False Data Injection & DoS Attack Simulation & Detection on the IEEE 14-Bus System

This project simulates and detects **False Data Injection (FDI)** and **Denial-of-Service (DoS)** attacks on a smart grid using the IEEE 14-bus system. It features a full pipeline from simulation in MATLAB Simulink, real-time UDP-based data communication, Man-in-the-Middle (MITM) attack injection, live dashboard visualization, and machine learning-based FDI detection.

## 🧠 Project Overview

The project is focused on cyber-physical security for smart grids, particularly targeting integrity and availability attacks:

- **Simulation**: MATLAB Simulink model of the IEEE 14-bus system generates real-time 3-phase voltage and current measurements.
- **MITM Proxy**: A Python-based proxy intercepts the data stream to inject FDI attacks and simulate DoS by packet corruption.
- **Dashboard**: Live dashboard displays voltage/current data, attack alerts, and model predictions.
- **Detection**: A machine learning model trained on the simulated data detects FDI attacks with high accuracy.

---

## 📁 Repository Structure

```
.
├── Fourteen_bus_data.slx      # MATLAB Simulink files for IEEE 14-bus system
├── udp_send.py                # Sends UDP packets to the dashboard
├── mitm.py                    # Executes MITM logic for FDI and DoS attacks
├── mitm_data.py               # Generates labeled dataset from simulation
├── train_fdia_model.py        # Trains the FDI detection machine learning model
├── dashboard/                 # Contains HTML/CSS/JS for real-time monitoring dashboard
│   └── index.html             # Main visualization file
```

---

## 🚀 How to Run

Follow the steps below to simulate the system and run the detection and visualization pipeline:

### 1. Simulate the Grid in MATLAB Simulink

- Open the Simulink file  `Fourteen_bus_dataslx`
- Run the simulation. It sends voltage and current values for all 14 buses via UDP to `localhost:5005`.

### 2. Run the MITM Attack Proxy

In your terminal:

```bash
python mitm.py
```

This will:

- Intercept incoming packets on port 5005.
- Inject False Data or simulate DoS attacks.
- Forward the (potentially tampered) data to port 5006.

### 3. Launch the Server for Dashboard

Run:

```bash
python udp_send.py
```

This will:

- Receive data on port 5006.
- Log the incoming data and send it to the web dashboard.

### 4. Open the Dashboard

- Navigate to the `dashboard/` folder.
- Open `index.html` in a web browser.
- You will see real-time voltage/current values, attack detection alerts, and ML-based predictions.

---

## 🧪 FDI Detection Model

- To retrain the FDI model, run:

```bash
python train_fdia_model.py
```

- The model uses a Histogram-Based Gradient Boosting Classifier.
- Dataset is generated using `mitm_data.py`.

---

