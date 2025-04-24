import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io  

# Set page configuration
st.set_page_config(page_title="Reed-Frost Epidemic Model", layout="wide")

# Display title and description with equation at the top
st.title("Reed-Frost Epidemic Model Simulator")
st.markdown("""
This interactive application simulates the spread of infectious disease using the Reed-Frost chain binomial model.
Adjust the parameters using the sliders and see how they affect the epidemic curve and disease dynamics.

**Definition**: The Reed-Frost model is a discrete-time epidemic model that describes how a disease spreads through a population over distinct time periods or generations.

**Reed-Frost Equation**: 
$C_{t+1} = S_t \cdot (1 - (1-p)^{C_t})$ 

Where:
- $C_{t+1}$ is the number of new cases in the next time period
- $S_t$ is the number of susceptible individuals in the current time period
- $p$ is the probability of effective contact between an infected and susceptible individual
- $C_t$ is the number of cases in the current time period

**Parameters:**
- **P**: The probability of effective contact between infected and susceptible individuals
- **C0**: The initial number of cases in the population
- **S0**: The initial number of susceptible individuals
- **B**: Birth rate per time period (adds new susceptibles)
- **I**: Immigration rate per time period (adds new susceptibles)
- **D**: Death rate per time period (applies to all population groups)
- **M**: Mortality rate from disease (applies only to active cases)
""")

# Helper function for downloading plots
def get_image_download_link(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return buf

# Define the Reed-Frost model function
def reed_frost_model(p, c0, s0, b=0, i=0, d=0, m=0, time_periods=100):
    """
    Implement the Reed-Frost epidemic model
    
    Parameters:
    - p: Probability of effective contact
    - c0: Initial number of cases
    - s0: Initial number of susceptible individuals
    - b: Birth rate per time period
    - i: Immigration rate per time period
    - d: Death rate per time period
    - m: Mortality rate from disease per time period
    - time_periods: Number of time periods to simulate
    
    Returns:
    - DataFrame with cases, susceptible, and immune counts over time
    """
    # Initialize arrays to store results
    time = np.arange(time_periods)
    cases = np.zeros(time_periods)
    susceptible = np.zeros(time_periods)
    immune = np.zeros(time_periods)
    total = np.zeros(time_periods)
    
    # Set initial conditions
    cases[0] = c0
    susceptible[0] = s0
    immune[0] = 0
    total[0] = c0 + s0
    
    # Run the model
    for t in range(1, time_periods):
        # Calculate new cases based on Reed-Frost equation
        q = (1 - p) ** cases[t-1]  # Probability of escaping infection
        new_cases = susceptible[t-1] * (1 - q)
        
        # Update population with births, deaths, immigration
        new_births = total[t-1] * b
        new_immigrants = i
        natural_deaths = total[t-1] * d
        disease_deaths = cases[t-1] * m
        
        # Update states
        cases[t] = new_cases
        susceptible[t] = susceptible[t-1] - new_cases + new_births + new_immigrants
        if total[t-1] > 0:
            susceptible[t] -= (susceptible[t-1]/total[t-1]) * natural_deaths
        
        immune[t] = immune[t-1] + cases[t-1] - disease_deaths
        if total[t-1] > 0:
            immune[t] -= (immune[t-1]/total[t-1]) * natural_deaths
            
        total[t] = cases[t] + susceptible[t] + immune[t]
        
    # Create DataFrame
    df = pd.DataFrame({
        'Time': time,
        'Cases': cases,
        'Susceptible': susceptible,
        'Immune': immune,
        'Total': total
    })
    
    return df

# Create sidebar with parameters
st.sidebar.header("Model Parameters")

p = st.sidebar.slider("Probability of Effective Contact (P)", 0.0, 1.0, 0.1, 0.01)
c0 = st.sidebar.number_input("Initial number of cases (C0)", 1, 1000, 1)
s0 = st.sidebar.number_input("Initial number of susceptible individuals (S0)", 1, 10000, 10)
b = st.sidebar.slider("Birth rate per time period (B)", 0.0, 0.2, 0.0, 0.01)
i = st.sidebar.slider("Immigration rate per time period (I)", 0, 10, 0, 1)
d = st.sidebar.slider("Death rate per time period (D)", 0.0, 0.2, 0.0, 0.01)
m = st.sidebar.slider("Mortality rate from disease per time period (M)", 0.0, 0.5, 0.0, 0.01)
time_periods = st.sidebar.slider("Number of time periods", 10, 200, 50, 10)

# Run the model with the current parameters
results = reed_frost_model(p, c0, s0, b, i, d, m, time_periods)

# Display summary statistics ABOVE the tabs
st.header("Summary Statistics")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Peak Cases", f"{results['Cases'].max():.2f}")
with col2:
    st.metric("Time of Peak", f"{results['Cases'].idxmax()} generations")
with col3:
    st.metric("Final Susceptible", f"{results['Susceptible'].iloc[-1]:.2f}")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["Epidemic Curve", "Sensitivity Analysis", "Parameter Relationships", "Data Table"])

with tab1:
    st.header("Epidemic Curve")
    
    # Display the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results['Time'], results['Cases'], label='Active Cases', color='red', linewidth=2)
    ax.plot(results['Time'], results['Susceptible'], label='Susceptible', color='blue', linewidth=2)
    ax.plot(results['Time'], results['Immune'], label='Immune', color='green', linewidth=2)
    ax.plot(results['Time'], results['Total'], label='Total Population', color='black', linestyle='--', linewidth=1)
    
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
    st.write(f"Maximum daily incidence: {np.diff(np.append(0, (results['Immune'] + results['Cases']).values)).max():.2f}")
    st.write(f"Final population size: {results['Total'].iloc[-1]:.2f}")
