import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io

def run():
    # Display title and description
    st.title("🪳 Leslie Matrix Cockroach Population Model")
    st.markdown("#### 📈 A discrete, age-structured model of population growth")
    st.markdown("""
    This interactive application simulates cockroach population dynamics using a Leslie Matrix model based on 
    population biology principles for studying vector and pest populations.
    Adjust the parameters using the sliders and see how they affect the population growth.
    
    **Definition**: The Leslie Matrix model is a discrete-time, age-structured population model that describes 
    population growth when age-specific survival rates and reproductive rates can be estimated. It's a fundamental 
    mathematical tool in population ecology, representing stage transitions and fecundity in a matrix format.
    
    **Core Concept**: At the heart of the Leslie Matrix model is the projection of current population structure 
    to future time steps through matrix multiplication. The matrix contains survival probabilities on the sub-diagonal 
    and fecundity values in the first row.
    
    **Leslie Matrix Equation**:
    $n_{t+1} = L n_t$
    
    Where:
    - $n_{t+1}$ is the population vector at the next time step
    - $L$ is the Leslie Matrix containing survival rates and fecundity values
    - $n_t$ is the current population vector by age class
    
    **The Model's Logic**:
    1. Each age class has a specific survival probability to the next age class
    2. Only certain adult age classes reproduce, with age-specific fecundity
    3. The population structure changes over time until reaching a stable age distribution
    4. The matrix eigenvalue determines whether the population grows, shrinks, or stabilizes
    
    **Parameters:**
    - **Egg survival rate**: Daily survival probability for eggs in the oothecae (egg cases)
    - **Nymphal survival rate**: Daily survival probability for nymphs (immature cockroaches)
    - **Adult survival rate**: Daily survival probability for adult cockroaches
    - **Initial population**: Starting number of individuals
    - **Fecundity values**: Number of eggs produced at different adult ages
    """)
    
    # Create sidebar with parameters
    st.sidebar.header("Model Parameters")
    
    # Survival rates
    st.sidebar.subheader("Survival Rates (p)")
    st.sidebar.markdown("""
    *In population models, daily survivorship (p) is a key parameter influencing 
    population growth and structure. Small changes in survival rates can have significant
    effects on long-term population outcomes.*
    """)
    
    egg_survival = st.sidebar.slider("Egg daily survival rate", 0.0, 1.0, 0.9, 0.01, 
                                    help="Survival rates for eggs typically range from 0.85-0.95 in protected oothecae")
    nymphal_survival = st.sidebar.slider("Nymphal daily survival rate", 0.0, 1.0, 0.9, 0.01,
                                       help="Nymphs typically have mobility but are vulnerable to predation and control measures")
    adult_survival = st.sidebar.slider("Adult daily survival rate", 0.0, 1.0, 0.9, 0.01, 
                                      help="Adult cockroaches tend to have high survival rates in favorable environments")
    
    # Initial population
    st.sidebar.subheader("Population Parameters")
    initial_population = st.sidebar.number_input("Initial population (adults at day 8)", 1, 10000, 1, 
                                               help="Starting population of adult cockroaches")
    
    # Fecundity values
    st.sidebar.subheader("Fecundity Values (f)")
    st.sidebar.markdown("""
    *Cockroaches produce oothecae (egg cases) at specific intervals during adulthood.
    Each ootheca contains multiple eggs, with reproduction occurring in distinct pulses.*
    """)
    
    fecundity_1 = st.sidebar.number_input("Fecundity at first oviposit (day 12)", 0, 500, 50, 
                                         help="Number of eggs produced at first reproduction")
    fecundity_2 = st.sidebar.number_input("Fecundity at second oviposit (day 17)", 0, 500, 50, 
                                         help="Number of eggs produced at second reproduction")
    fecundity_3 = st.sidebar.number_input("Fecundity at third oviposit (day 22)", 0, 500, 50, 
                                         help="Number of eggs produced at third reproduction")
    fecundity_4 = st.sidebar.number_input("Fecundity at fourth oviposit (day 27)", 0, 500, 50, 
                                         help="Number of eggs produced at fourth reproduction")
    
    # Time periods to simulate
    num_days = st.sidebar.slider("Number of days to simulate", 28, 200, 60, 
                               help="Length of simulation in days")
    
    # Developmental stages for cockroaches
    st.sidebar.subheader("Life Stage Durations")
    st.sidebar.markdown("""
    *Cockroaches have specific developmental periods for each life stage,
    with transitions between stages that are crucial for understanding population dynamics.*
    """)
    
    egg_stage_duration = st.sidebar.slider("Egg stage duration (days)", 1, 30, 7, 
                                         help="Duration of egg development inside ootheca")
    nymphal_stage_duration = st.sidebar.slider("Nymphal stage duration (days)", 1, 90, 11, 
                                             help="Duration of nymphal stage before reaching adulthood")
    
    # Create a helper function to detect development stages
    def get_stage(day):
        if day < egg_stage_duration:
            return "Egg"
        elif day < egg_stage_duration + nymphal_stage_duration:
            return "Nymph"
        else:
            return "Adult"
    
    def run_leslie_model(egg_survival, nymphal_survival, adult_survival, 
                        initial_pop, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
                        num_days, egg_stage_duration, nymphal_stage_duration):
        """
        Run the Leslie Matrix population model simulation for cockroaches
        
        Parameters:
        - egg_survival: Daily survival rate for eggs
        - nymphal_survival: Daily survival rate for nymphs
        - adult_survival: Daily survival rate for adults
        - initial_pop: Initial adult population
        - fecundity_1-4: Number of eggs laid at each oviposition event
        - num_days: Number of days to simulate
        - egg_stage_duration: Number of days in egg stage
        - nymphal_stage_duration: Number of days in nymphal stage
        
        Returns:
        - Population matrix and summary data
        """
        adult_stage_start = egg_stage_duration + nymphal_stage_duration
        total_stages = max(28, adult_stage_start + 20)  # Ensure we have enough stages for development
        
        # Create the Leslie Matrix
        leslie_matrix = np.zeros((total_stages, total_stages))
        
        # Set survival probabilities (subdiagonal)
        for i in range(total_stages-1):
            if i < egg_stage_duration:  # Egg stage
                leslie_matrix[i+1, i] = egg_survival
            elif i < egg_stage_duration + nymphal_stage_duration:  # Nymphal stage
                leslie_matrix[i+1, i] = nymphal_survival
            else:  # Adult stage
                leslie_matrix[i+1, i] = adult_survival
        
        # Set fecundity values (first row)
        # Reproduction occurs on specific adult days (counted from adult emergence)
        reproduction_days = [
            12,  # First oviposition
            17,  # Second oviposition
            22,  # Third oviposition
            27   # Fourth oviposition
        ]
        
        # Ensure indices are valid
        for i, day in enumerate(reproduction_days):
            if adult_stage_start + (day - adult_stage_start) < total_stages:
                if day >= adult_stage_start:
                    leslie_matrix[0, day] = [fecundity_1, fecundity_2, fecundity_3, fecundity_4][i]
        
        # Initialize population vector
        population = np.zeros(total_stages)
        population[adult_stage_start] = initial_pop  # Start with initial adults
        
        # Initialize results matrix
        results = np.zeros((num_days, total_stages))
        results[0, :] = population
        
        # Run the simulation
        for t in range(1, num_days):
            population = leslie_matrix @ population
            results[t, :] = population
        
        return results, total_stages, egg_stage_duration, nymphal_stage_duration
    
    # Run the model
    results, total_stages, egg_stage_duration, nymphal_stage_duration = run_leslie_model(
        egg_survival, nymphal_survival, adult_survival, 
        initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
        num_days, egg_stage_duration, nymphal_stage_duration
    )
    
    # Process results for visualization
    days = np.arange(1, num_days+1)
    egg_indices = range(0, egg_stage_duration)
    nymph_indices = range(egg_stage_duration, egg_stage_duration + nymphal_stage_duration)
    adult_indices = range(egg_stage_duration + nymphal_stage_duration, total_stages)
    
    # Calculate life stage totals
    eggs = np.sum(results[:, egg_indices], axis=1)
    nymphs = np.sum(results[:, nymph_indices], axis=1)
    adults = np.sum(results[:, adult_indices], axis=1)
    total_population = eggs + nymphs + adults
    
    # Calculate stage percentages
    percent_eggs = np.zeros(num_days)
    percent_nymphs = np.zeros(num_days)
    percent_adults = np.zeros(num_days)
    
    for i in range(num_days):
        if total_population[i] > 0:
            percent_eggs[i] = (eggs[i] / total_population[i]) * 100
            percent_nymphs[i] = (nymphs[i] / total_population[i]) * 100
            percent_adults[i] = (adults[i] / total_population[i]) * 100
    
    # Create a DataFrame for the summary data
    summary_df = pd.DataFrame({
        'Day': days,
        'Eggs': eggs,
        'Nymphs': nymphs,
        'Adults': adults,
        'Total': total_population,
        '%Eggs': percent_eggs,
        '%Nymphs': percent_nymphs,
        '%Adults': percent_adults
    })
    
    # Display the overall statistics
    st.header("Population Summary")
    
    # Add interpretation about pest population dynamics
    intrinsic_growth_rate = np.log(total_population[-1] / initial_population) / num_days if total_population[-1] > 0 else 0
    
    st.info(f"""
    **Population Growth Analysis:**
    
    The estimated intrinsic growth rate (r) of this cockroach population is {intrinsic_growth_rate:.4f}, 
    representing the per capita rate of increase. This growth rate is determined by the interaction of 
    stage-specific survival rates and reproductive patterns.
    
    For pest species like cockroaches, population models help identify critical life stages and 
    thresholds for effective control measures. The exponential growth potential shown here 
    demonstrates why early intervention is essential for pest management.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Number - Eggs", f"{eggs[-1]:.0f}")
        
    with col2:
        st.metric("Final Number - Nymphs", f"{nymphs[-1]:.0f}")
        
    with col3:
        st.metric("Final Number - Adults", f"{adults[-1]:.0f}")
        
    with col4:
        st.metric("Total Final Population", f"{total_population[-1]:.0f}")
    
    # Helper function to convert figure to downloadable data
    def fig_to_bytes(fig):
        """Convert a matplotlib figure to bytes for downloading"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        return buf
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Population Trends", "Stage Distribution", "Age Structure", "Data Table"])
    
    with tab1:
        st.header("Population Growth Over Time")
        
        # FIRST show the plots
        # Population trend plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(days, eggs, label='Eggs', color='#f7d060', linewidth=2)
        ax1.plot(days, nymphs, label='Nymphs', color='#ff6e40', linewidth=2)
        ax1.plot(days, adults, label='Adults', color='#5d5d5d', linewidth=2)
        ax1.plot(days, total_population, label='Total', color='#1e88e5', linewidth=3, linestyle='--')
        
        ax1.set_xlabel('Day', fontsize=12)
        ax1.set_ylabel('Number of Individuals', fontsize=12)
        ax1.set_title('Cockroach Population Growth by Life Stage', fontsize=14)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Use log scale if the population gets very large
        if max(total_population) > 10000:
            ax1.set_yscale('log')
            st.info("Note: Using logarithmic scale for y-axis due to large population numbers")
        
        st.pyplot(fig1)
        
        # Streamlit download button
        st.download_button(
            label="Download Population Trends Plot",
            data=fig_to_bytes(fig1),
            file_name="cockroach_population_trends.png",
            mime="image/png"
        )
        
        # Population growth rate plot
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        growth_rates = np.zeros(num_days)
        for i in range(1, num_days):
            growth_rates[i] = (total_population[i] / total_population[i-1] - 1) * 100 if total_population[i-1] > 0 else 0
        
        ax2.plot(days[1:], growth_rates[1:], color='#2ca02c', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Day', fontsize=12)
        ax2.set_ylabel('Growth Rate (%)', fontsize=12)
        ax2.set_title('Daily Population Growth Rate', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        st.pyplot(fig2)
        
        st.download_button(
            label="Download Growth Rate Plot",
            data=fig_to_bytes(fig2),
            file_name="cockroach_growth_rate.png",
            mime="image/png"
        )
        
        # THEN add interpretation text
        # UNIQUE INTERPRETATION FOR POPULATION TRENDS TAB - COCKROACH FOCUS
        st.markdown("""
        **Cockroach Population Growth Dynamics:**
        
        This visualization reveals characteristic exponential growth patterns in cockroach populations, 
        similar to the patterns discussed in population biology literature on pest species. Several important 
        features are evident:
        
        1. **Pulse Reproduction Pattern**: Unlike continuous reproduction, cockroaches show distinct pulses of 
           reproduction as females produce oothecae at regular intervals, creating the step-like increases in 
           egg numbers. This pulse pattern is similar to the reproduction patterns observed in gonotrophic cycles 
           of some vector species.
        
        2. **Stage Duration Effects**: The relative lengths of egg and nymphal stages create a time-delayed 
           pattern of population growth, with waves of individuals moving through developmental stages. The 
           longer the nymphal stage, the greater the delay between reproduction and the emergence of new 
           reproductive adults.
        
        3. **Control Implications**: For pest management, this growth curve suggests that interventions targeting 
           adults before they reproduce will be most effective at preventing population explosions. The 
           exponential nature of the growth curve demonstrates why early detection and intervention are crucial.
        """)
        
        # UNIQUE INTERPRETATION FOR GROWTH RATE - COCKROACH FOCUS
        mean_growth = np.mean(growth_rates[1:])
        st.markdown(f"""
        **Growth Rate Pattern Analysis:**
        
        The growth rate plot reveals how cockroach populations expand over time. The mean daily growth rate 
        of {mean_growth:.2f}% translates to significant population increases over short periods.
        
        Key observations:
        
        1. **Cyclical Growth Patterns**: The spikes in growth rate correspond to the maturation of large 
           cohorts of nymphs into reproductive adults, followed by the production of new egg cases. These 
           cycles create predictable windows for control interventions.
        
        2. **Initial Establishment Phase**: Cockroach infestations typically show a lag phase with lower 
           growth rates followed by a rapid acceleration as multiple reproducing females become established. 
           This pattern mirrors the logistic growth equation described in population ecology literature.
        
        3. **Carrying Capacity Effects**: In natural settings, growth rates would eventually decline as 
           populations approach carrying capacity due to resource limitations and density-dependent factors. 
           In human habitations, this ceiling may be much higher than in natural environments.
        """)
