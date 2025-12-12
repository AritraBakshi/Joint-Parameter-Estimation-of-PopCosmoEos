# Joint Parameter Estimation of PopCosmoEoS

This repository contains code, notebooks, and outputs for performing **joint parameter estimation** for *Population, Cosmology, and Equation-of-State (EoS)* modeling.  
It includes both pre– and post–“q” analyses, Python scripts for running parameter simulations, and generated plots (corner plots, trace plots, etc.).

---

## 📁 Repository Structure
```bash
├── LICENSE
├── notebooks
│ ├── after_q
│ │ └── pce_model_dmlq_cleaned.ipynb
│ └── before_q
│ └── pce_model_dml_cleaned.ipynb
└── with q
├── corner plot_100.png
├── pce_100.py
└── traceplot_100.png
```
### **notebooks/**
Contains cleaned Jupyter notebooks for running and exploring the PCE (PopCosmoEoS) models.

- **before_q/**  
  Notebook for baseline parameter estimation prior to incorporating the *q-parameter*.

- **after_q/**  
  Notebook for updated modeling that includes *q* in the estimation pipeline.

### **with q/**
Contains final Python scripts and output visualizations for the q-enabled analysis.

- **pce_100.py** – Script used for running inference/parameter estimation  
- **corner plot_100.png** – Posterior corner plot  
- **traceplot_100.png** – Sampler trace plot

---

## 🚀 Getting Started

### **1. Clone the repository**
```bash
git clone https://github.com/AritraBakshi/Joint-Parameter-Estimation-of-PopCosmoEos.git
cd Joint-Parameter-Estimation-of-PopCosmoEoS
```
### **2. Install dependencies**
```bash
pip install -r requirements.txt
```
If requirements.txt does not exist, you may generate it:

```bash
pip freeze > requirements.txt
```
### **3. Run the model**
Run the main script:

```bash
python "with q/pce_100.py"
```
Or open any notebook:

```bash
jupyter notebook
```
---
### **📊 Outputs**
Outputs generated using the estimation:

1. Corner plot – Posterior distribution visualization

2. Trace plot – Shows sampler convergence and mixing

---
🧠 Background / Methodology

This project explores joint estimation of astrophysical/cosmological population parameters and EoS properties, incorporating a q-dependent modification to the likelihood or population model.

The methodology involves:
```bash
MCMC or nested sampling
        |
Hierarchical modeling
        |
Population likelihood evaluation
        |
Cosmological priors
        | 
EoS constraints
```
---

### **📘 Notebooks Overview**
Notebook	Description :

* pce_model_dml_cleaned.ipynb	Baseline model without q-parameter
* pce_model_dmlq_cleaned.ipynb	Updated model with q-parameter

---
### **📄 License**
This project is distributed under the terms of the LICENSE file in this repository.
