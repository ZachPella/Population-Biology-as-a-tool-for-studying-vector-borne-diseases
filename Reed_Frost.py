import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io  

def run():
        # Display title and description with equation at the top
    st.title("🧬 Reed-Frost Chain Binomial Model")
    st.markdown("#### 🔄 Generation-based epidemic simulator")
    st.markdown("""
    ### Introduction to the Reed-Frost Model
    
    This interactive application simulates the spread of infectious disease using the Reed-Frost chain binomial model. 
    Adjust the parameters using the sliders and see how they affect the epidemic curve and disease dynamics.
    
    **Definition**: The Reed-Frost model is a discrete-time epidemic model that describes how a disease spreads through a population over distinct time periods or generations. It is a fundamental mathematical tool in epidemiology, representing a simplified framework for understanding disease transmission dynamics.
    
    **Core Concept**: At the heart of the Reed-Frost model is the probability of effective contact (P) between infected and susceptible individuals. For some diseases, this probability is high (e.g., influenza), while for others it is low (e.g., leprosy) or requires special forms of contact (e.g., HIV).
    
    **Reed-Frost Equation**: 
    $C_{t+1} = S_t \cdot (1 - (1-p)^{C_t})$ 
    
    Where:
    - $C_{t+1}$ is the number of new cases in the next time period
    - $S_t$ is the number of susceptible individuals in the current time period
    - $p$ is the probability of effective contact between an infected and susceptible individual
    - $C_t$ is the number of cases in the current time period
    
    **The Model's Logic**: 
    1. Each infectious individual makes contact with every other individual in the population during a time period
    2. The probability that a contact between an infectious and susceptible individual will lead to a new infection is P
    3. The probability that a susceptible escapes infection from any single case is (1-P)
    4. The probability that a susceptible escapes infection from all cases is $(1-P)^{C_t}$
    5. Therefore, the probability of becoming infected is $1-(1-P)^{C_t}$
    
    ### Adjustable Parameters:
    - **P**: The probability of effective contact between infected and susceptible individuals
    - **C0**: The initial number of cases in the population
    - **S0**: The initial number of susceptible individuals
    - **B**: Birth rate per time period (adds new susceptibles)
    - **I**: Immigration rate per time period (adds new susceptibles)
    - **D**: Death rate per time period (applies to all population groups)
    - **M**: Mortality rate from disease (applies only to active cases)
    
    This model was originally developed to understand epidemics in closed populations, but can be modified to account for population dynamics like births, deaths, and immigration, allowing for the study of endemic diseases and more complex transmission patterns.
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
    tab1, tab2, tab3, tab4 = st.tabs(["Epidemic Curve", "Sensitivity Analysis", "Parameter Relationships", "Epidemic Parameters"])
    
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
        
                # Add epidemic curve interpretation
        st.subheader("Interpretation")
        
        # Calculate key epidemic metrics for interpretation
        peak_time = results['Cases'].idxmax()
        peak_value = results['Cases'].max()
        epidemic_duration = results['Cases'][results['Cases'] > 0.01].count()
        final_immune = results['Immune'].iloc[-1]
        final_susceptible = results['Susceptible'].iloc[-1]
        initial_population = results['Total'].iloc[0]
        
        # Calculate attack rate (percentage of initial population that gets infected)
        attack_rate = (final_immune / initial_population) * 100
        
        st.write("""
        ### The Reed-Frost Chain Binomial Model
        
        The Reed-Frost model is a discrete-time (generation-based) epidemic model that describes how disease spreads through a population in distinct time periods. The epidemic curve illustrates the core components of the model:
        
        **Susceptible Population (Blue)**: Individuals who have not yet been infected but are capable of becoming infected if they come into contact with the disease.
        
        **Active Cases (Red)**: Infected individuals who can transmit the disease to susceptibles. The Reed-Frost model assumes these individuals contact everyone in the population during their infectious period.
        
        **Immune Population (Green)**: Individuals who have recovered from infection and acquired immunity, preventing reinfection.
        
        **Key Equation**: $C_{t+1} = S_t \cdot (1 - (1-p)^{C_t})$, where $C_{t+1}$ is new cases in the next time period, $S_t$ is susceptibles in the current time period, $p$ is probability of effective contact, and $C_t$ is current cases.
        """)
        
        st.write("""
        ### Dynamics of the Epidemic Curve
        
        **Probability of Effective Contact (P={})**: This parameter significantly impacts disease transmission. Higher P values lead to faster spread and larger outbreaks.
        
        **Outbreak Size**: With the current parameters, approximately {}% of the initial population became infected, with peak cases of {} occurring at generation {}.
        
        **Threshold Effects**: The model demonstrates that below certain thresholds of P, epidemics may not occur at all, while higher values cause rapid spread.
        
        **Population Dynamics**: With birth rate (B={}) and immigration (I={}), new susceptibles enter the population, potentially sustaining transmission longer than in a closed population.
        
        **Immunity Effects**: As the susceptible population (blue) declines and immune population (green) increases, the epidemic naturally slows due to reduced availability of susceptible individuals.
        """.format(p, round(attack_rate, 2), round(peak_value, 2), peak_time, b, i))
        
        # Add interpretation based on current parameter values
        if p < 0.05:
            transmission_text = "At low probability of effective contact (P={:.2f}), each infected individual generates fewer than one new case on average, leading to a small outbreak that quickly dies out.".format(p)
        elif p < 0.2:
            transmission_text = "With moderate probability of effective contact (P={:.2f}), the epidemic grows steadily before declining as immunity builds in the population.".format(p)
        else:
            transmission_text = "With high probability of effective contact (P={:.2f}), disease spreads rapidly, reaches a high peak, and affects a large proportion of the susceptible population.".format(p)
        
        if b > 0 or i > 0:
            demographic_text = "The addition of new susceptibles through births (B={:.2f}) and/or immigration (I={:.2f}) affects the epidemic pattern, potentially allowing for endemic disease to establish or multiple waves to occur.".format(b, i)
        else:
            demographic_text = "In this closed population (no births or immigration), the epidemic follows a classic single-wave pattern that eventually burns out once sufficient immunity is achieved."
        
        if d > 0 or m > 0:
            mortality_text = "Population loss through natural deaths (D={:.2f}) and/or disease mortality (M={:.2f}) reduces both susceptible and immune populations, affecting the final epidemic size and potentially allowing for resurgence if immunity is lost.".format(d, m)
        else:
            mortality_text = "With no population loss (no deaths), the total population remains constant, and the final state represents the complete redistribution of individuals between susceptible and immune compartments."
        
        
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
            ### Probability of Effective Contact (P)
            
            The sensitivity analysis demonstrates how the **probability of effective contact (P)** is a critical determinant of epidemic dynamics. As shown in the Reed-Frost equation $C_{t+1} = S_t \\cdot (1 - (1-p)^{C_t})$, P directly determines how many susceptibles become infected in each generation.
            
            **Key observations from the model:**
            - At P values below 0.1, epidemics may not establish (or "take off")
            - As P increases from 0.1 to 0.5, both the peak height and total cases increase rapidly
            - At high P values (>0.5), the epidemic curve shifts left, reaching peak faster
            - The final size of the epidemic is highly sensitive to even small changes in P
            
            **Mathematical insight**: The nonlinear relationship between P and total cases emerges from the compounding effect of multiple generations of transmission. Small reductions in P are amplified over successive infection cycles.
            """)
        elif param_name == "c0":
            st.write("""
            ### Initial Cases (C0)
            
            The **initial number of cases (C0)** represents the seed of the epidemic. The Reed-Frost model demonstrates that while C0 affects the timing of the epidemic, it has limited impact on the final epidemic size in closed populations.
            
            **Key observations from the model:**
            - Higher C0 values cause the epidemic to reach its peak earlier
            - Final epidemic size remains nearly constant regardless of C0, provided P is sufficient for epidemic spread
            - In deterministic models, even a single case (C0=1) will cause an epidemic if P is above threshold
            - The timing shift occurs because more initial cases create more infections in the first generation
                        
            **Mathematical insight**: The Reed-Frost equation shows that timing shifts occur because a larger C0 value increases the initial force of infection $(1-(1-P)^{C_t})$, but the overall epidemic trajectory follows similar dynamics once established.
            """)
        elif param_name == "s0":
            st.write("""
            ### Initial Susceptible Population (S0)
            
            The **initial susceptible population (S0)** directly determines the potential size of the epidemic. The Reed-Frost model demonstrates a proportional relationship between S0 and both peak cases and total infected.
            
            **Key observations from the model:**
            - Total cases scale almost linearly with S0 when other parameters remain constant
            - Peak height increases proportionally with S0
            - The shape of the epidemic curve remains similar, just scaled by population size
            - Time to peak is only slightly affected by S0 (larger populations peak marginally later)
                        
            **Mathematical insight**: In the Reed-Frost equation, S0 appears as a direct multiplier of new cases. However, the actual fraction of susceptibles who become infected depends on the probability of effective contact (P) and the overall population dynamics.
            """)
        elif param_name == "b":
            st.write("""
            ### Birth Rate (B)
            
            The **birth rate parameter (B)** adds new susceptibles to the population over time, potentially sustaining transmission beyond a single epidemic wave. This moves the model from a simple closed-population epidemic to one that can model endemic disease patterns.
            
            **Key observations from the model:**
            - With B=0, the epidemic follows a single wave pattern that eventually burns out
            - As B increases, the tail of the epidemic curve extends, creating potential for multiple waves
            - At high enough birth rates, a steady endemic state may establish
            - New susceptibles extend the duration of the epidemic rather than increasing its peak height
                        
            **Mathematical insight**: When birth rate is incorporated, the Reed-Frost model can demonstrate oscillatory behavior or steady endemic states similar to what's seen in real-world endemic diseases like measles, which show periodic epidemic cycles driven by the accumulation of new susceptible individuals.
            """)
        elif param_name == "i":
            st.write("""
            ### Immigration Rate (I)
            
            The **immigration parameter (I)** introduces new susceptibles from outside the population, which can sustain transmission similarly to births. This parameter is particularly relevant for modeling geographically connected populations.
            
            **Key observations from the model:**
            - Immigration creates a continuous inflow of susceptibles that can maintain transmission
            - Unlike birth rate which scales with population size, immigration is typically modeled as a constant inflow
            - Even low immigration rates can prevent epidemic burnout if P is sufficient
            - Immigration can trigger new epidemic waves after an initial wave has subsided
                        
            **Mathematical insight**: Immigration differs from birth rate in that immigrants can enter already infected, creating new transmission chains even in largely immune populations. This makes immigration particularly important in metapopulation models that consider connected geographic regions.
            """)
        elif param_name == "d":
            st.write("""
            ### Death Rate (D)
            
            The **death rate parameter (D)** removes individuals from all compartments (susceptible, infected, immune), affecting population dynamics and potentially disease transmission.
            
            **Key observations from the model:**
            - Higher death rates reduce the total population size over time
            - Death reduces the number of susceptibles, potentially limiting epidemic size
            - However, deaths also remove immune individuals, potentially increasing vulnerability to subsequent waves
            - The overall effect depends on the interaction between death rate and other demographic parameters
                        
            **Mathematical insight**: Death rate creates a flow out of all compartments, while birth/immigration create flows into the susceptible compartment. The balance between these flows determines whether the epidemic is single-wave, endemic, or cyclic.
            """)
        elif param_name == "m":
            st.write("""
            ### Mortality Rate from Disease (M)
            
            The **disease mortality parameter (M)** specifically removes individuals from the infected compartment, representing case fatality from the disease.
            
            **Key observations from the model:**
            - Disease mortality reduces the number of infected individuals who transition to immunity
            - Higher disease mortality can slightly reduce transmission by shortening the infectious period
            - The effect on epidemic dynamics is usually small unless M is very high
            - Disease mortality primarily affects public health impact rather than transmission dynamics
                        
            **Mathematical insight**: In the Reed-Frost model, disease mortality affects how many cases move from the infected to removed compartment due to death rather than recovery. This parameter is particularly important for diseases like Ebola with high case fatality rates, where mortality can significantly affect transmission dynamics.
            """)
        else:
            st.write(f"""
            ### {param_to_vary}
            
            This parameter affects the epidemic dynamics by changing how individuals move between the susceptible, infected, and immune compartments in the Reed-Frost model. The sensitivity analysis shows how variations in this parameter affect both the epidemic curve shape and final outcome.
            
            **Key observations from the model:**
            - Changes in this parameter create nonlinear effects in epidemic outcomes
            - The relationship shown in the graph demonstrates potential thresholds or tipping points
            - Both peak timing and final size are affected
                        
            **Mathematical insight**: The Reed-Frost model demonstrates how even simple epidemic models can exhibit complex behaviors due to the interactions between parameters. This underscores the importance of sensitivity analysis in understanding epidemic dynamics.
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
            ax4.set_title(title, fontsize=10)
            
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
                ### Probability of Effective Contact (P) × Initial Susceptible Population (S₀)
            
                **Key observations:**
            
                The probability of effective contact (P) and the initial susceptible population (S₀) interact in ways that directly relate to the Reed-Frost equation $C_{t+1} = S_t \\cdot (1 - (1-p)^{C_t})$. The heatmap reveals:
            
                - **Epidemic Threshold**: At low P values (<0.1), epidemics fail to establish regardless of S₀, creating a clear threshold boundary visible on the heatmap
                - **Nonlinear Interactions**: The relationship between P and S₀ is not simply additive - increasing both parameters simultaneously creates a stronger effect than would be predicted by changing each parameter individually
                - **Diminishing Returns**: At very high P values, further increases have minimal effect on outcomes when S₀ is limited
                - **Scale Dependence**: The epidemic threshold for P decreases as S₀ increases, meaning larger populations can sustain epidemics at lower contact probabilities
            
                **Mathematical insights:**
            
                These interactions demonstrate fundamental principles of epidemic theory:
            
                - The basic reproductive number (R₀) in the Reed-Frost model is related to both P and S₀
                - The threshold effect occurs when P reaches a value where each case generates at least one new case on average
                - The contour lines on the heatmap represent combinations of P and S₀ that produce equal epidemic sizes""")
                    
            elif (x_name == "p" and y_name == "c0") or (x_name == "c0" and y_name == "p"):
                st.write("""
                ### Probability of Effective Contact (P) × Initial Cases (C₀)
            
                **Key observations:**
            
                The heatmap reveals important interactions between the probability of effective contact (P) and the initial number of cases (C₀):
            
                - **Threshold Effects**: At low P values, increasing C₀ has minimal impact on epidemic outcomes
                - **Critical Points**: As P approaches the epidemic threshold, the number of initial cases becomes more influential
                - **Saturation Effects**: At high P values, even a single initial case (C₀=1) leads to a full epidemic
                - **Time-Shifting**: While not visible on the heatmap, higher C₀ values primarily affect epidemic timing rather than final size
            
                **Mathematical insights:**
            
                The Reed-Frost equation $C_{t+1} = S_t \\cdot (1 - (1-p)^{C_t})$ shows that C₀ appears as an exponent in the first generation calculation. This means:
            
                - The initial force of infection $(1-(1-P)^{C_0})$ increases nonlinearly with C₀
                - At low P values, multiple initial cases are needed to overcome stochastic extinction
                - At high P values, the term $(1-P)^{C_t}$ quickly approaches zero even with small C₀""")
                    
            elif (x_name == "b" and y_name == "p") or (x_name == "p" and y_name == "b"):
                st.write("""
                ### Birth Rate (B) × Probability of Effective Contact (P)
            
                **Key observations:**
            
                The birth rate (B) and probability of effective contact (P) interact to determine whether a disease remains endemic:
            
                - **Endemic Thresholds**: At low P values, even high birth rates cannot sustain transmission
                - **Critical Zone**: The heatmap reveals a transition zone where the disease shifts from epidemic to endemic behavior
                - **Oscillatory Patterns**: In certain combinations of B and P, the disease may show cyclical patterns (not visible in the static heatmap)
                - **Intensity Gradient**: Higher values of both parameters lead to greater disease burden, but through different mechanisms
            
                **Mathematical insights:**
            
                In the modified Reed-Frost model that includes births:
            
                - Birth rate determines the inflow of new susceptibles, which can prevent the susceptible population from being depleted
                - Endemic equilibrium occurs when new infections balance new susceptibles entering the population
                - The interplay between P and B determines whether R₀ remains above 1 in the long term""")
                    
            elif (x_name == "b" and y_name == "d") or (x_name == "d" and y_name == "b"):
                st.write("""
                ### Birth Rate (B) × Death Rate (D)
            
                **Key observations:**
            
                The birth rate (B) and death rate (D) together determine population dynamics and disease patterns:
            
                - **Population Stability**: When B≈D, the population size remains relatively stable
                - **Growing vs. Shrinking**: When B>D, the population grows, creating more susceptibles over time; when B<D, the population shrinks
                - **Turnover Effects**: Even with stable population size (B≈D), high values of both parameters create rapid population turnover
                - **Immunity Persistence**: High death rates can reduce population immunity by removing immune individuals
            
                **Mathematical insights:**
            
                In population models:
            
                - The difference (B-D) determines net population growth
                - Population turnover (min(B,D)) affects how quickly the population composition changes
                - In a growing population, the effective reproduction number may increase over time as susceptibles accumulate""")
            else:
                st.write(f"""
                ### {x_param} × {y_param} Interaction
            
                **Key observations:**
            
                This heatmap reveals how {metric.lower()} changes as a function of both {x_param.lower()} and {y_param.lower()}:
            
                - **Parameter Sensitivity**: The gradient patterns show which parameter has stronger influence on outcomes
                - **Interaction Zones**: Regions of rapid color change indicate critical combinations where small parameter shifts cause large outcome changes
                - **Threshold Effects**: Clear boundaries between light and dark regions may indicate epidemic thresholds
                - **Optimum Combinations**: The contour patterns reveal combinations of parameters that produce equivalent outcomes
            
                **Mathematical insights:**
            
                The Reed-Frost model demonstrates that:
            
                - Disease transmission is inherently nonlinear, creating complex parameter interactions
                - Some parameter combinations create compensatory effects, where changes in one parameter can offset changes in another
                - The contour lines represent sets of parameter values that yield equal epidemic outcomes""")
    with tab4:
        st.header("Epidemic Parameters")
        
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
    
if __name__ == "__main__":
    run()
