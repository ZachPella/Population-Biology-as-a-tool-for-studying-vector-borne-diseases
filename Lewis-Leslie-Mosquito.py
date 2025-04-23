import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt

# Set page configuration
st.set_page_config(page_title="Leslie Matrix Population Model", layout="wide")

# Display title and description
st.title("Leslie Matrix Population Model Simulator")
st.markdown("""
This interactive tool simulates population growth using a Leslie Matrix model,
commonly used in population ecology to track age-structured populations over time.

Adjust the parameters using the sliders to see how they affect population dynamics.
""")

# Create sidebar with parameters
st.sidebar.header("Life History Parameters")

# Survival rates
egg_survival = st.sidebar.slider("Egg daily survival rate", 0.0, 1.0, 0.9, 0.01)
larval_survival = st.sidebar.slider("Larval daily survival rate", 0.0, 1.0, 0.9, 0.01)
adult_survival = st.sidebar.slider("Adult daily survival rate", 0.0, 1.0, 0.8, 0.01)

# Initial population
initial_population = st.sidebar.number_input("Initial population (adults)", 1, 10000, 1000)

# Fecundity values
fecundity_1 = st.sidebar.number_input("Fecundity at first oviposition (day 12)", 0, 500, 120)
fecundity_2 = st.sidebar.number_input("Fecundity at second oviposition (day 17)", 0, 500, 100)
fecundity_3 = st.sidebar.number_input("Fecundity at third oviposition (day 22)", 0, 500, 80)
fecundity_4 = st.sidebar.number_input("Fecundity at fourth oviposition (day 27)", 0, 500, 60)

# Time periods to simulate
num_days = st.sidebar.slider("Number of days to simulate", 28, 180, 60)
num_stages = 28  # We use 28 age classes: egg, larvae, and adults by day

def run_leslie_model(egg_survival, larval_survival, adult_survival, 
                    initial_pop, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
                    num_days, num_stages=28):
    """
    Run the Leslie Matrix population model simulation
    
    Parameters:
    - egg_survival: Daily survival rate for eggs
    - larval_survival: Daily survival rate for larvae
    - adult_survival: Daily survival rate for adults
    - initial_pop: Initial adult population
    - fecundity_1-4: Number of eggs laid at each oviposition event
    - num_days: Number of days to simulate
    - num_stages: Number of age classes (default: 28)
    
    Returns:
    - Population matrix and summary data
    """
    # Create the Leslie Matrix
    leslie_matrix = np.zeros((num_stages, num_stages))
    
    # Set survival probabilities (subdiagonal)
    for i in range(num_stages-1):
        if i < 7:  # Egg stage (days 0-7)
            leslie_matrix[i+1, i] = egg_survival
        elif i < 11:  # Larval stage (days 8-11)
            leslie_matrix[i+1, i] = larval_survival
        else:  # Adult stage (days 12+)
            leslie_matrix[i+1, i] = adult_survival
    
    # Set fecundity values (first row)
    leslie_matrix[0, 11] = fecundity_1  # First oviposition at day 12
    leslie_matrix[0, 16] = fecundity_2  # Second oviposition at day 17
    leslie_matrix[0, 21] = fecundity_3  # Third oviposition at day 22
    leslie_matrix[0, 26] = fecundity_4  # Fourth oviposition at day 27
    
    # Initialize population vector (adults on day 8)
    population = np.zeros(num_stages)
    population[8] = initial_pop
    
    # Initialize results matrix to store population at each time step
    results = np.zeros((num_days, num_stages))
    results[0, :] = population
    
    # Run the simulation
    for day in range(1, num_days):
        population = leslie_matrix @ population
        results[day, :] = population
    
    return results

# Run the simulation
results = run_leslie_model(
    egg_survival, larval_survival, adult_survival, 
    initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
    num_days
)

# Process results
days = np.arange(num_days)
eggs = np.sum(results[:, 0:8], axis=1)  # Sum of stages 0-7
larvae = np.sum(results[:, 8:12], axis=1)  # Sum of stages 8-11
adults = np.sum(results[:, 12:], axis=1)  # Sum of stages 12+
total = eggs + larvae + adults

# Create a DataFrame for the summary data
summary_df = pd.DataFrame({
    'Day': days,
    'Eggs': eggs,
    'Larvae': larvae,
    'Adults': adults,
    'Total': total,
    '%Eggs': eggs / total * 100,
    '%Larvae': larvae / total * 100,
    '%Adults': adults / total * 100
})

# Display the overall statistics
st.header("Population Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Final Number - Eggs", f"{eggs[-1]:.0f}")
with col2:
    st.metric("Final Number - Larvae", f"{larvae[-1]:.0f}")
with col3:
    st.metric("Final Number - Adults", f"{adults[-1]:.0f}")

st.metric("Total Population at End", f"{total[-1]:.0f}")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Population Trends", "Stage Distribution", "Data Table"])

with tab1:
    st.header("Population Growth Over Time")
    
    # Create line chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, eggs, label='Eggs', color='#f7d060')
    ax.plot(days, larvae, label='Larvae', color='#ff6e40')
    ax.plot(days, adults, label='Adults', color='#5d5d5d')
    ax.plot(days, total, label='Total', color='#1e88e5', linewidth=2)
    
    ax.set_xlabel('Days')
    ax.set_ylabel('Number of Individuals')
    ax.set_title('Population Growth by Life Stage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Use log scale if the population gets very large
    if max(total) > 10000:
        ax.set_yscale('log')
        st.info("Note: Using logarithmic scale for y-axis due to large population numbers")
    
    st.pyplot(fig)

with tab2:
    st.header("Population Composition")
    
    # Create a stacked area chart
    chart_data = pd.DataFrame({
        'Day': days,
        'Eggs': eggs,
        'Larvae': larvae,
        'Adults': adults
    })
    
    # Reshape for Altair
    chart_data_melted = pd.melt(
        chart_data, 
        id_vars=['Day'], 
        value_vars=['Eggs', 'Larvae', 'Adults'],
        var_name='Stage', 
        value_name='Count'
    )
    
    # Create stacked area chart
    chart = alt.Chart(chart_data_melted).mark_area().encode(
        x='Day:Q',
        y=alt.Y('Count:Q', stack='normalize'),
        color=alt.Color('Stage:N', scale=alt.Scale(
            domain=['Eggs', 'Larvae', 'Adults'],
            range=['#f7d060', '#ff6e40', '#5d5d5d']
        ))
    ).properties(
        width=700,
        height=400,
        title='Relative Proportion of Life Stages Over Time'
    )
    
    st.altair_chart(chart, use_container_width=True)
    
    # Create a pie chart for the final day
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    final_values = [eggs[-1], larvae[-1], adults[-1]]
    labels = ['Eggs', 'Larvae', 'Adults']
    colors = ['#f7d060', '#ff6e40', '#5d5d5d']
    ax2.pie(final_values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax2.set_title(f'Population Composition on Day {num_days-1}')
    st.pyplot(fig2)

with tab3:
    st.header("Detailed Data")
    
    # Show the Leslie matrix
    st.subheader("Leslie Matrix")
    leslie_matrix = np.zeros((28, 28))
    
    # Set survival probabilities
    for i in range(27):
        if i < 7:
            leslie_matrix[i+1, i] = egg_survival
        elif i < 11:
            leslie_matrix[i+1, i] = larval_survival
        else:
            leslie_matrix[i+1, i] = adult_survival
    
    # Set fecundity values
    leslie_matrix[0, 11] = fecundity_1
    leslie_matrix[0, 16] = fecundity_2
    leslie_matrix[0, 21] = fecundity_3
    leslie_matrix[0, 26] = fecundity_4
    
    # Create a heatmap of the Leslie matrix
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    im = ax3.imshow(leslie_matrix, cmap='viridis')
    ax3.set_title('Leslie Matrix Visualization')
    ax3.set_xlabel('Current Stage (day)')
    ax3.set_ylabel('Next Stage (day)')
    fig3.colorbar(im, ax=ax3, label='Transition Rate')
    st.pyplot(fig3)
    
    # Display the full results table
    st.subheader("Population Data")
    
    # Format the table to show key stages by day
    detailed_df = pd.DataFrame(results, columns=[f'Day {i}' for i in range(28)])
    detailed_df.index = [f'Day {i}' for i in range(num_days)]
    
    # Add summary columns
    detailed_df['Eggs'] = eggs
    detailed_df['Larvae'] = larvae
    detailed_df['Adults'] = adults
    detailed_df['Total'] = total
    
    st.dataframe(detailed_df)
    
    # Download buttons
    csv = summary_df.to_csv(index=False)
    st.download_button(
        label="Download Summary Data (CSV)",
        data=csv,
        file_name="leslie_matrix_summary.csv",
        mime="text/csv"
    )
    
    csv_detailed = detailed_df.to_csv()
    st.download_button(
        label="Download Detailed Data (CSV)",
        data=csv_detailed,
        file_name="leslie_matrix_detailed.csv",
        mime="text/csv"
    )

# Explanation of the model
st.header("About the Leslie Matrix Model")
st.markdown("""
The Leslie matrix model is a discrete, age-structured model of population growth that projects the population of each age class forward in time. It incorporates:

1. **Age-specific survival rates**: The probability that an individual of a given age will survive to the next age class
2. **Age-specific fecundity rates**: The number of offspring produced by individuals in each age class

### Structure of the Leslie Matrix:
- Each row and column represents an age class
- The first row contains fecundity values (number of offspring produced)
- The subdiagonal contains survival probabilities (probability of surviving to the next age class)

### How it works:
- The population at time t+1 is calculated by multiplying the Leslie matrix by the population vector at time t
- Over time, the population structure approaches a stable age distribution
- The dominant eigenvalue of the Leslie matrix gives the long-term growth rate of the population

In this specific implementation, we track:
- Egg stage (days 0-7)
- Larval stage (days 8-11)
- Adult stage (days 12+)
- Reproduction events at specific adult days (12, 17, 22, 27)

This model is particularly useful for understanding the dynamics of insect populations, wildlife management, and conservation biology.
""")
