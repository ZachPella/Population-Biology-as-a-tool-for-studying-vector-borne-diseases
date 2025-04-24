import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io
import math

# Set page configuration
st.set_page_config(page_title="Macdonald's Model of Vectorial Capacity", layout="wide")

# Display title and description
st.title("Macdonald's Model of Vectorial Capacity")
st.markdown("""
This interactive application simulates the vectorial capacity in vector-borne disease transmission using Macdonald's model.
Adjust the parameters using the sliders and see how they affect the vectorial capacity and disease transmission potential.

**Definition**: Vectorial capacity is the average number of potentially infective bites that would eventually arise from all the vectors that bite a single infectious host on a single day.

**Macdonald's Equation**: 
$C = \\frac{ma^2b p^n}{-\\ln(p)}$ 

**Parameters:**
- **Vector:host ratio (m)**: Number of vectors per human host
- **Biting rate (a)**: Number of bites per vector per day on hosts
- **Vector competence (b)**: Proportion of vectors that develop infection after feeding on an infectious host
- **Daily survival probability (p)**: Probability that a vector survives one day
- **Extrinsic incubation period (n)**: Number of days required for the pathogen to develop in the vector
""")

# Create sidebar with parameters
st.sidebar.header("Model Parameters")

# Vector density parameters
vector_host_ratio = st.sidebar.slider("Vector:host ratio (m)", 0.1, 50.0, 10.0, 0.1, 
                                    help="Number of vectors per human host")

# Biting behavior parameters
biting_rate = st.sidebar.slider("Daily biting rate (a)", 0.0, 1.0, 0.3, 0.01, 
                             help="Probability that a vector feeds on a host in one day")

# Host preference calculation
st.sidebar.subheader("Host Preference Factors")
host_preference = st.sidebar.slider("Host preference index (0-1)", 0.0, 1.0, 0.5, 0.01,
                                 help="Preference for human hosts vs other animals (1 = feeds only on humans)")

days_between_feedings = st.sidebar.number_input("Number of days between feedings", 1, 14, 3, 
                                            help="Average number of days between blood meals")

# Vector competence parameter
vector_competence = st.sidebar.slider("Vector competence (b)", 0.0, 1.0, 0.5, 0.01,
                                   help="Proportion of vectors that develop infection after feeding on an infectious host")

# Survival parameters
daily_survival = st.sidebar.slider("Daily survival probability (p)", 0.0, 1.0, 0.9, 0.01,
                                help="Probability that a vector survives one day")

# Calculate vector lifespan after EIP
vector_lifespan = 1/(-math.log(daily_survival)) if daily_survival > 0 else 0
st.sidebar.text(f"Vector lifespan: {vector_lifespan:.2f} days")

# Pathogen parameters
extrinsic_incubation = st.sidebar.slider("Extrinsic incubation period (n)", 1, 30, 10, 1,
                                      help="Days required for pathogen development in vector")

# Function to calculate vectorial capacity
def calculate_vectorial_capacity(m, a, b, p, n):
    """
    Calculate the vectorial capacity using Macdonald's formula
    
    Parameters:
    - m: Vector:host ratio
    - a: Daily biting rate
    - b: Vector competence
    - p: Daily survival probability
    - n: Extrinsic incubation period
    
    Returns:
    - C: Vectorial capacity
    """
    # Prevent domain errors with invalid parameters
    if p <= 0 or p >= 1:
        return 0
    
    # Handle other potential errors
    try:
        # Macdonald's formula: C = ma²bp^n/-ln(p)
        return (m * (a**2) * b * (p**n)) / (-math.log(p))
    except Exception:
        # Return 0 if any calculation error occurs
        return 0

# Calculate vectorial capacity with current parameters
vectorial_capacity = calculate_vectorial_capacity(
    vector_host_ratio,
    biting_rate,
    vector_competence,
    daily_survival,
    extrinsic_incubation
)

# Calculate basic reproduction number (R0) assuming human recovery rate of 0.14 (~ 7 days infectious period)
recovery_rate = 0.14  # recovery rate per day
r0 = vectorial_capacity * vector_competence / recovery_rate

# Calculate minimum survival rate needed to maintain transmission
if biting_rate > 0 and vector_competence > 0 and vector_host_ratio > 0:
    critical_daily_survival = math.exp(-1/extrinsic_incubation)
else:
    critical_daily_survival = 0

# Display the summary statistics under the header section
st.header("Summary Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Vectorial Capacity (C)", f"{vectorial_capacity:.4f}")
    
with col2:
    st.metric("Basic Reproduction Number (R₀)", f"{r0:.4f}", 
             help="R₀ = Vectorial Capacity × Vector Competence ÷ Human Recovery Rate")
    
with col3:
    st.metric("Critical Daily Survival", f"{critical_daily_survival:.4f}",
             delta=f"{(daily_survival - critical_daily_survival):.4f}", 
             delta_color="normal",
             help="Minimum daily survival rate needed for vectors to live long enough to transmit the pathogen")

# Helper function to convert figure to downloadable data
def fig_to_bytes(fig):
    """Convert a matplotlib figure to bytes for downloading"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    return buf

# Create tabs for different analyses
tab1, tab2, tab3, tab4= st.tabs(["Sensitivity Analysis", "Host Preference Impact", "Parameter Relationships", "Data Table"])

with tab1:
    st.header("Epidemic Curve")
    
    # First show summary statistics in a row of metrics ABOVE the plot
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Peak Cases", f"{results['Cases'].max():.2f}")
    with col2:
        st.metric("Time of Peak", f"{results['Cases'].idxmax()} generations")
    with col3:
        st.metric("Attack Rate", f"{(results['Immune'].iloc[-1] / (s0 + c0)) * 100:.1f}%")
    with col4:
        st.metric("Final Susceptible", f"{results['Susceptible'].iloc[-1]:.2f}")
    
    # Then display the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['Time'], results['Cases'], label='Active Cases', color='red', linewidth=2)
    ax.plot(results['Time'], results['Susceptible'], label='Susceptible', color='blue', linewidth=2)
    ax.plot(results['Time'], results['Immune'], label='Immune', color='green', linewidth=2)
    ax.plot(results['Time'], results['Total'], label='Total Population', color='black', linestyle='--', linewidth=1)
    
    # Identify epidemic phases
    peak_day = results['Cases'].idxmax()
    early_phase = int(peak_day * 0.3) if peak_day > 0 else 0
    late_phase = min(len(results) - 1, int(peak_day * 1.7)) if peak_day > 0 else len(results) - 1
    
    # Highlight phases with different background colors
    if peak_day > 0:
        ax.axvspan(0, early_phase, alpha=0.2, color='green', label='Early Phase')
        ax.axvspan(early_phase, peak_day, alpha=0.2, color='yellow', label='Growth Phase')
        ax.axvspan(peak_day, late_phase, alpha=0.2, color='blue', label='Decline Phase')
        ax.axvspan(late_phase, results['Time'].max(), alpha=0.2, color='purple', label='Late Phase')
    
    ax.set_xlabel('Time (Generations)', fontsize=12)
    ax.set_ylabel('Number of Individuals', fontsize=12)
    ax.set_title('Reed-Frost Epidemic Curve', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # Add download button for the plot
    st.download_button(
        label="Download Plot",
        data=get_image_download_link(fig),
        file_name="epidemic_curve.png",
        mime="image/png"
    )

with tab2:
    st.header("Sensitivity Analysis")
    
    # Choose parameter to vary
    param_to_vary = st.selectbox(
        "Select parameter to vary:",
        ["Probability of Effective Contact (P)", "Initial Cases (C0)", 
         "Initial Susceptible (S0)", "Birth Rate (B)", "Immigration Rate (I)",
         "Death Rate (D)", "Disease Mortality Rate (M)"]
    )
    
    # Set up parameter ranges based on selection
    if param_to_vary == "Probability of Effective Contact (P)":
        param_range = np.linspace(0.01, 1.0, 50)
        param_name = "p"
        x_label = "Probability of Effective Contact"
    elif param_to_vary == "Initial Cases (C0)":
        param_range = np.arange(1, 51)
        param_name = "c0"
        x_label = "Initial Cases"
    elif param_to_vary == "Initial Susceptible (S0)":
        param_range = np.linspace(10, 500, 50)
        param_name = "s0"
        x_label = "Initial Susceptible Population"
    elif param_to_vary == "Birth Rate (B)":
        param_range = np.linspace(0, 0.2, 50)
        param_name = "b"
        x_label = "Birth Rate per Time Period"
    elif param_to_vary == "Immigration Rate (I)":
        param_range = np.arange(0, 21)
        param_name = "i"
        x_label = "Immigration Rate per Time Period"
    elif param_to_vary == "Death Rate (D)":
        param_range = np.linspace(0, 0.2, 50)
        param_name = "d"
        x_label = "Death Rate per Time Period"
    else:  # Disease Mortality Rate (M)
        param_range = np.linspace(0, 0.5, 50)
        param_name = "m"
        x_label = "Disease Mortality Rate per Time Period"
    
    # Calculate metrics for each parameter value
    peak_cases = []
    total_infected = []
    
    for param_val in param_range:
        # Set up parameters dictionary
        params = {
            "p": p,
            "c0": c0,
            "s0": s0,
            "b": b,
            "i": i,
            "d": d,
            "m": m
        }
        # Update with parameter to vary
        params[param_name] = param_val
        
        # Run model with these parameters
        results_temp = reed_frost_model(
            params["p"], params["c0"], params["s0"], 
            params["b"], params["i"], params["d"], params["m"], 
            time_periods
        )
        
        # Calculate metrics
        peak_cases.append(results_temp['Cases'].max())
        total_infected.append(results_temp['Immune'].iloc[-1] + results_temp['Cases'].iloc[-1])
    
    # Create tabs for different metrics to display
    metric_tab1, metric_tab2 = st.tabs(["Peak Cases", "Total Infected"])
    
    with metric_tab1:
        # Create figure for peak cases
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(param_range, peak_cases, 'r-', linewidth=2)
        
        # Add vertical line for current parameter value
        if param_name == "p":
            current_val = p
        elif param_name == "c0":
            current_val = c0
        elif param_name == "s0":
            current_val = s0
        elif param_name == "b":
            current_val = b
        elif param_name == "i":
            current_val = i
        elif param_name == "d":
            current_val = d
        else:  # m
            current_val = m
        
        # Find the current peak cases value
        current_idx = np.abs(param_range - current_val).argmin()
        current_peak = peak_cases[current_idx]
        
        ax1.axvline(x=current_val, color='black', linestyle='--', alpha=0.7)
        ax1.plot(current_val, current_peak, 'ro', markersize=8)
        ax1.annotate(f'Current value: {current_val:.2f}\nPeak: {current_peak:.2f}', 
                    xy=(current_val, current_peak), xytext=(10, -30),
                    textcoords='offset points', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel('Peak Number of Cases', fontsize=12)
        ax1.set_title(f'Effect of {param_to_vary} on Peak Cases', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        st.pyplot(fig1)
    
    with metric_tab2:
        # Create figure for total infected
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(param_range, total_infected, 'b-', linewidth=2)
        
        # Find the current total infected value
        current_idx = np.abs(param_range - current_val).argmin()
        current_total = total_infected[current_idx]
        
        ax2.axvline(x=current_val, color='black', linestyle='--', alpha=0.7)
        ax2.plot(current_val, current_total, 'bo', markersize=8)
        ax2.annotate(f'Current value: {current_val:.2f}\nTotal: {current_total:.2f}', 
                    xy=(current_val, current_total), xytext=(10, -30),
                    textcoords='offset points', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        ax2.set_xlabel(x_label, fontsize=12)
        ax2.set_ylabel('Total Number of Infected', fontsize=12)
        ax2.set_title(f'Effect of {param_to_vary} on Total Infected', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
    
    # Add interpretation
    st.subheader("Interpretation")
    
    if param_name == "p":
        st.write("""
        **Probability of Effective Contact (P)** has a significant impact on epidemic dynamics. As P increases, 
        both peak cases and total infected increase. When P is small, the epidemic may not take off at all.
        
        **Control implications**: Reducing the probability of effective contact through interventions like social distancing, 
        masks, or hand hygiene can effectively reduce both peak and total cases.
        """)
    elif param_name == "c0":
        st.write("""
        **Initial Cases (C0)** primarily affects the timing of the epidemic rather than its final size. With more initial cases, 
        the peak occurs earlier but reaches approximately the same height. The total number infected is relatively insensitive 
        to initial cases.
        
        **Control implications**: Early detection and isolation of initial cases can delay the epidemic but may not substantially 
        reduce its final size unless combined with other interventions.
        """)
    elif param_name == "s0":
        st.write("""
        **Initial Susceptible Population (S0)** strongly affects both peak height and total cases. 
        Larger susceptible populations lead to larger epidemics with higher peaks.
        
        **Control implications**: Reducing the susceptible population through vaccination or creating immune subpopulations 
        through targeted protection can effectively reduce epidemic potential.
        """)
    elif param_name == "b":
        st.write("""
        **Birth Rate (B)** adds new susceptibles to the population over time. Higher birth rates can sustain transmission 
        for longer periods and may lead to endemic disease patterns rather than a single epidemic wave.
        
        **Control implications**: In populations with high birth rates, continuous control measures may be needed to prevent 
        resurgence as new susceptibles enter the population.
        """)
    else:
        st.write(f"""
        **{param_to_vary}** affects the epidemic dynamics by changing how quickly individuals move between the susceptible, 
        infected, and recovered compartments. This parameter influences both the peak height and the total number of cases.
        
        **Control implications**: Understanding how this parameter affects disease spread can help design targeted interventions 
        to mitigate outbreaks.
        """)

with tab3:
    st.header("Parameter Relationships")
    
    # Select parameters for X and Y axes
    col1, col2 = st.columns(2)
    
    with col1:
        x_param = st.selectbox(
            "Select parameter for X-axis:",
            ["Probability of Effective Contact (P)", "Initial Cases (C0)", 
             "Initial Susceptible (S0)", "Birth Rate (B)"],
            index=0
        )
    
    with col2:
        y_param = st.selectbox(
            "Select parameter for Y-axis:",
            ["Probability of Effective Contact (P)", "Initial Cases (C0)", 
             "Initial Susceptible (S0)", "Birth Rate (B)"],
            index=2
        )
    
    # Ensure different parameters are selected
    if x_param == y_param:
        st.error("Please select different parameters for X and Y axes.")
    else:
        # Define parameter mappings
        param_configs = {
            "Probability of Effective Contact (P)": {
                "range": np.linspace(0.01, 0.5, 15),
                "name": "p",
                "label": "Probability of Effective Contact"
            },
            "Initial Cases (C0)": {
                "range": np.arange(1, 16),
                "name": "c0",
                "label": "Initial Cases"
            },
            "Initial Susceptible (S0)": {
                "range": np.linspace(10, 200, 15),
                "name": "s0",
                "label": "Initial Susceptible Population"
            },
            "Birth Rate (B)": {
                "range": np.linspace(0, 0.2, 15),
                "name": "b",
                "label": "Birth Rate per Time Period"
            }
        }
        
        # Create parameter grids
        x_range = param_configs[x_param]["range"]
        y_range = param_configs[y_param]["range"]
        x_name = param_configs[x_param]["name"]
        y_name = param_configs[y_param]["name"]
        
        # Create meshgrid for heatmap
        X, Y = np.meshgrid(x_range, y_range)
        peak_Z = np.zeros_like(X)
        total_Z = np.zeros_like(X)
        
        # Calculate metrics for each parameter combination
        for i in range(len(y_range)):
            for j in range(len(x_range)):
                # Set default parameters
                params = {
                    "p": p,
                    "c0": c0,
                    "s0": s0,
                    "b": b,
                    "i": i,
                    "d": d,
                    "m": m
                }
                
                # Update with grid values
                params[x_name] = X[i, j]
                params[y_name] = Y[i, j]
                
                # Run model with these parameters
                results_temp = reed_frost_model(
                    params["p"], params["c0"], params["s0"], 
                    params["b"], params["i"], params["d"], params["m"], 
                    time_periods
                )
                
                # Store metrics
                peak_Z[i, j] = results_temp['Cases'].max()
                total_Z[i, j] = results_temp['Immune'].iloc[-1] + results_temp['Cases'].iloc[-1]
        
        # Create selector for metric to display
        metric = st.radio(
            "Select metric to display:",
            ["Peak Cases", "Total Infected"],
            horizontal=True
        )
        
        # Choose data based on selected metric
        if metric == "Peak Cases":
            Z = peak_Z
            cmap = 'viridis'
            title = f'Peak Cases as a Function of {x_param} and {y_param}'
        else:  # Total Infected
            Z = total_Z
            cmap = 'plasma'
            title = f'Total Infected as a Function of {x_param} and {y_param}'
        
        # Create heatmap
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        contour = ax4.contourf(X, Y, Z, 20, cmap=cmap)
        
        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax4)
        cbar.set_label(metric, fontsize=12)
        
        # Mark current parameter values if within range
        current_x = params[x_name]
        current_y = params[y_name]
        
        if (min(x_range) <= current_x <= max(x_range)) and (min(y_range) <= current_y <= max(y_range)):
            ax4.plot(current_x, current_y, 'ro', markersize=10)
            ax4.annotate('Current values', xy=(current_x, current_y), xytext=(10, 10),
                        textcoords='offset points', ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        ax4.set_xlabel(param_configs[x_param]["label"], fontsize=12)
        ax4.set_ylabel(param_configs[y_param]["label"], fontsize=12)
        ax4.set_title(title, fontsize=14)
        
        st.pyplot(fig4)
        
        # Download button for the plot
        st.download_button(
            label="Download Parameter Relationship Plot",
            data=get_image_download_link(fig4),
            file_name=f"reed_frost_{x_name}_{y_name}_{metric.lower().replace(' ', '_')}.png",
            mime="image/png"
        )
        
        # Add explanation
        st.subheader("Parameter Interaction Effects")
        
        if (x_name == "p" and y_name == "s0") or (x_name == "s0" and y_name == "p"):
            st.write("""
            **Key observations:**
            
            The probability of effective contact (P) and the initial susceptible population (S₀) interact to determine 
            epidemic dynamics. Areas with darker colors indicate conditions more favorable for disease spread.
            
            The interaction between these parameters is important for understanding epidemic thresholds - at low values 
            of both parameters, epidemics may not occur, while high values of both lead to large outbreaks.
            
            **Control implications:**
            
            Both reducing contact rates and reducing the susceptible population (e.g., through vaccination) can help 
            control disease spread.
            
            The contour map shows that multiple combinations of parameters can achieve the same outcome, allowing for 
            flexible control strategies.
            """)
        else:
            st.write(f"""
            **Key observations:**
            
            This heatmap shows how {metric.lower()} changes as a function of both {x_param.lower()} and {y_param.lower()}.
            
            Darker colors indicate conditions more favorable for disease spread, while lighter colors represent conditions 
            less conducive to transmission.
            
            The steepness of the gradient indicates the sensitivity of the outcome to changes in each parameter.
            
            **Control implications:**
            
            Understanding parameter interactions helps identify the most effective combination of interventions.
            
            Some parameter combinations may have synergistic effects on reducing disease spread.
            """)

with tab4:
    st.header("Data Table")
    
    # Show the data table with highlighting
    st.dataframe(results.style.highlight_max(axis=0, color='yellow'))
    
    # Add download button for CSV
    csv = results.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name="reed_frost_results.csv",
        mime="text/csv"
    )
    
    # Show summary statistics
    st.subheader("Summary Statistics")
    st.write(f"Peak number of cases: {results['Cases'].max():.2f} at time {results['Cases'].idxmax()}")
    st.write(f"Total individuals who became infected: {results['Immune'].iloc[-1] + results['Cases'].iloc[-1]:.2f}")
    st.write(f"Final susceptible population: {results['Susceptible'].iloc[-1]:.2f}")
    st.write(f"Attack rate: {((results['Immune'].iloc[-1] + results['Cases'].iloc[-1]) / (s0 + c0)) * 100:.2f}%")
    st.write(f"Maximum daily incidence: {np.diff(np.append(0, (results['Immune'] + results['Cases']).values)).max():.2f}")
    st.write(f"Final population size: {results['Total'].iloc[-1]:.2f}")
