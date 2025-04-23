import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(page_title="Reed-Frost Epidemic Model", layout="wide")

# Display title and description
st.title("Reed-Frost Epidemic Model Simulator")
st.markdown("""
This interactive tool simulates the spread of an infectious disease using the Reed-Frost model.
Adjust the parameters using the sliders and see how they affect the epidemic curve.

**Parameters:**
- **P**: The probability of effective contact between infected and susceptible individuals
- **C0**: The initial number of cases in the population
- **S0**: The initial number of susceptible individuals
- **B**: Birth rate per time period (adds new susceptibles)
- **I**: Immigration rate per time period (adds new susceptibles)
- **D**: Death rate per time period (applies to all population groups)
- **M**: Mortality rate from disease (applies only to active cases)
""")

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

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Epidemic Curve", "Data Table", "Model Information"])

with tab1:
    st.header("Epidemic Curve")
    
    # Create plot
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
    st.header("Results Table")
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

with tab3:
    st.header("About the Reed-Frost Model")
    st.markdown("""
    The Reed-Frost model is a mathematical model of infectious disease transmission developed by Lowell Reed and Wade Hampton Frost in the 1920s. It's a discrete-time chain binomial model that predicts how a disease will spread in a closed population.

    ### Key Assumptions:
    - The population is homogeneously mixed
    - All cases are equally infectious
    - The disease has a fixed duration
    - Recovery from the disease confers immunity
    
    ### The Core Equation:
    The probability that a susceptible individual escapes infection is:
    
    $q = (1 - p)^C$
    
    Where:
    - $q$ is the probability of escaping infection
    - $p$ is the probability of effective contact
    - $C$ is the number of infectious cases
    
    ### Extensions in This Simulator:
    This simulator extends the basic Reed-Frost model to include:
    - Birth rate (B): Adds new susceptible individuals
    - Immigration rate (I): Adds new susceptible individuals
    - Death rate (D): Removes individuals from all groups
    - Disease mortality rate (M): Removes individuals from the infectious group
    
    ### References:
    - Abbey, H. (1952). An examination of the Reed-Frost theory of epidemics. Human Biology, 24(3), 201-233.
    - Fine, P. E. (1977). A commentary on the mechanical analogue to the Reed-Frost epidemic model. American Journal of Epidemiology, 106(2), 87-100.
    """)

# Helper function for downloading plots
def get_image_download_link(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    return buf
