import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io

def run():
    # Display title and description
    st.title("🦟 Leslie Matrix Vector Population Model")
    st.markdown("#### 📈 A discrete, age-structured model of population growth")
    st.markdown("""
    This interactive application simulates vector population dynamics using a Leslie Matrix model based on 
    population biology principles presented by Black & Moore for studying vector-borne diseases.
    Adjust the parameters using the sliders and see how they affect the population growth.
    
    **Definition**: The Leslie Matrix model is a discrete-time, age-structured population model that describes 
    population growth when age-specific survival rates and reproductive rates can be estimated. This fundamental 
    mathematical tool in population ecology represents stage transitions and fecundity in a matrix format.
    
    **Core Concept**: As described by Black & Moore, this matrix approach projects the current population structure 
    to future time steps through matrix multiplication, where the matrix contains survival probabilities on the sub-diagonal 
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
    3. The population structure changes over time until reaching a stable age distribution (SAD)
    4. The matrix eigenvalue determines whether the population grows, shrinks, or stabilizes
    
    **Parameters:**
    - **Daily survival rate (p)**: Probability of surviving through one day for each life stage
    - **Initial population**: Starting number of individuals
    - **Fecundity values**: Number of eggs produced at specific adult ages
    """)
    
    # Create sidebar with parameters
    st.sidebar.header("Model Parameters")
    
    # Survival rates
    st.sidebar.subheader("Survival Rates (p)")
    st.sidebar.markdown("""
    *According to Black & Moore, daily survivorship (p) is a key parameter influencing 
    population growth and structure. Small changes in survival rates can have significant
    effects on vectorial capacity.*
    """)
    
    egg_survival = st.sidebar.slider("Egg daily survival rate", 0.0, 1.0, 0.5, 0.01, 
                                    help="Survival rates for eggs in protected conditions")
    immature_survival = st.sidebar.slider("Immature daily survival rate", 0.0, 1.0, 0.6, 0.01,
                                       help="Survival rates for immature stages (larvae, pupae, nymphs)")
    adult_survival = st.sidebar.slider("Adult daily survival rate", 0.0, 1.0, 0.9, 0.01, 
                                      help="According to Table 24.1, adult survival rate strongly affects vectorial capacity")
    
    # Initial population
    st.sidebar.subheader("Population Parameters")
    initial_population = st.sidebar.number_input("Initial adult population", 1, 10000, 1, 
                                               help="Starting population of adults")
    
    # Fecundity values
    st.sidebar.subheader("Fecundity Values (f)")
    st.sidebar.markdown("""
    *As shown in Table 24.2 and Figure 24.7, vectors have different patterns of fecundity.
    Some species (like mosquitoes) have multiple gonotrophic cycles with pulse egg production.*
    """)
    
    fecundity_pattern = st.sidebar.selectbox(
        "Fecundity pattern",
        ["Multiple gonotrophic cycles (Fig 24.7A)", "Continuous production (Fig 24.7B)", "Single batch (Fig 24.7C)"]
    )
    
    if fecundity_pattern == "Multiple gonotrophic cycles (Fig 24.7A)":
        cycle_length = st.sidebar.slider("Days between gonotrophic cycles", 1, 14, 5, 
                                        help="According to the mosquito example, gonotrophic cycles require about 5 days")
        fecundity_1 = st.sidebar.number_input("Fecundity at first gonotrophic cycle", 0, 500, 120, 
                                            help="In the model mosquito, first cycle produces 120 eggs")
        fecundity_decline = st.sidebar.slider("Fecundity decline rate per cycle (%)", 0, 50, 20, 
                                            help="According to the mosquito example, fecundity declines in subsequent cycles")
    elif fecundity_pattern == "Continuous production (Fig 24.7B)":
        cycle_length = 1  # Daily production
        fecundity_1 = st.sidebar.number_input("Daily egg production", 0, 100, 5, 
                                            help="Continuous low-level egg production")
        fecundity_decline = 0
    else:  # Single batch
        cycle_length = 0  # No repeating cycles
        fecundity_1 = st.sidebar.number_input("Total batch size", 0, 5000, 1000, 
                                            help="According to hard tick example, can produce 1000-5000 eggs")
        adult_lifespan = st.sidebar.slider("Adult days until oviposition", 1, 30, 20, 
                                         help="Days until the single egg batch is produced")
        fecundity_decline = 100  # Complete decline after first batch
    
    # Time periods to simulate
    num_days = st.sidebar.slider("Number of days to simulate", 28, 200, 60, 
                               help="Length of simulation in days")
    
    # Developmental stages for vector
    st.sidebar.subheader("Life Stage Durations")
    st.sidebar.markdown("""
    *Black & Moore emphasize that age classes must be of equal length (days)
    for the model to be accurate.*
    """)
    
    egg_stage_duration = st.sidebar.slider("Egg stage duration (days)", 1, 30, 7, 
                                         help="Days spent in egg stage")
    immature_stage_duration = st.sidebar.slider("Immature stage duration (days)", 1, 90, 11, 
                                              help="Days spent in immature stages")
    
    # Create a helper function to detect development stages
    def get_stage(day):
        if day < egg_stage_duration:
            return "Egg"
        elif day < egg_stage_duration + immature_stage_duration:
            return "Immature"
        else:
            return "Adult"
    
    def run_leslie_model(egg_survival, immature_survival, adult_survival, 
                        initial_pop, fecundity_values, 
                        num_days, egg_stage_duration, immature_stage_duration):
        """
        Run the Leslie Matrix population model simulation
        
        Parameters:
        - egg_survival: Daily survival rate for eggs
        - immature_survival: Daily survival rate for immatures
        - adult_survival: Daily survival rate for adults
        - initial_pop: Initial adult population
        - fecundity_values: Dictionary mapping adult days to fecundity values
        - num_days: Number of days to simulate
        - egg_stage_duration: Number of days in egg stage
        - immature_stage_duration: Number of days in immature stage
        
        Returns:
        - Population matrix and summary data
        """
        adult_stage_start = egg_stage_duration + immature_stage_duration
        total_stages = max(30, adult_stage_start + 30)  # Ensure we have enough stages for development
        
        # Create the Leslie Matrix - as described in Figure 24.8
        leslie_matrix = np.zeros((total_stages, total_stages))
        
        # Set survival probabilities (subdiagonal)
        for i in range(total_stages-1):
            if i < egg_stage_duration:  # Egg stage
                leslie_matrix[i+1, i] = egg_survival
            elif i < egg_stage_duration + immature_stage_duration:  # Immature stage
                leslie_matrix[i+1, i] = immature_survival
            else:  # Adult stage
                leslie_matrix[i+1, i] = adult_survival
        
        # Set fecundity values (first row)
        for day, fecundity in fecundity_values.items():
            if day < total_stages:
                leslie_matrix[0, day] = fecundity
        
        # Initialize population vector
        population = np.zeros(total_stages)
        if fecundity_pattern == "Single batch (Fig 24.7C)":
            # For hard ticks, start with adults about to lay eggs
            population[adult_stage_start + adult_lifespan - 1] = initial_pop
        else:
            # Start with initial adults
            population[adult_stage_start] = initial_pop
        
        # Initialize results matrix
        results = np.zeros((num_days, total_stages))
        results[0, :] = population
        
        # Run the simulation - matrix multiplication as in Figure 24.8
        for t in range(1, num_days):
            population = np.matmul(leslie_matrix, population)
            results[t, :] = population
        
        return results, total_stages, egg_stage_duration, immature_stage_duration
    
    # Set up fecundity values based on pattern
    fecundity_values = {}
    if fecundity_pattern == "Multiple gonotrophic cycles (Fig 24.7A)":
        adult_stage_start = egg_stage_duration + immature_stage_duration
        for i in range(4):  # 4 gonotrophic cycles, as shown in mosquito example
            day = adult_stage_start + (i * cycle_length)
            decline_factor = (100 - (i * fecundity_decline)) / 100
            fecundity_values[day] = fecundity_1 * decline_factor
    
    elif fecundity_pattern == "Continuous production (Fig 24.7B)":
        adult_stage_start = egg_stage_duration + immature_stage_duration
        for i in range(30):  # Continuous production over 30 days
            day = adult_stage_start + i
            fecundity_values[day] = fecundity_1
    
    else:  # Single batch
        adult_stage_start = egg_stage_duration + immature_stage_duration
        day = adult_stage_start + adult_lifespan - 1
        fecundity_values[day] = fecundity_1
    
    # Run the model
    results, total_stages, egg_stage_duration, immature_stage_duration = run_leslie_model(
        egg_survival, immature_survival, adult_survival, 
        initial_population, fecundity_values, 
        num_days, egg_stage_duration, immature_stage_duration
    )
    
    # Process results for visualization
    days = np.arange(1, num_days+1)
    egg_indices = range(0, egg_stage_duration)
    immature_indices = range(egg_stage_duration, egg_stage_duration + immature_stage_duration)
    adult_indices = range(egg_stage_duration + immature_stage_duration, total_stages)
    
    # Calculate life stage totals
    eggs = np.sum(results[:, egg_indices], axis=1)
    immatures = np.sum(results[:, immature_indices], axis=1)
    adults = np.sum(results[:, adult_indices], axis=1)
    total_population = eggs + immatures + adults
    
    # Calculate stage percentages
    percent_eggs = np.zeros(num_days)
    percent_immatures = np.zeros(num_days)
    percent_adults = np.zeros(num_days)
    
    for i in range(num_days):
        if total_population[i] > 0:
            percent_eggs[i] = (eggs[i] / total_population[i]) * 100
            percent_immatures[i] = (immatures[i] / total_population[i]) * 100
            percent_adults[i] = (adults[i] / total_population[i]) * 100
    
    # Create a DataFrame for the summary data
    summary_df = pd.DataFrame({
        'Day': days,
        'Eggs': eggs,
        'Immatures': immatures,
        'Adults': adults,
        'Total': total_population,
        '%Eggs': percent_eggs,
        '%Immatures': percent_immatures,
        '%Adults': percent_adults
    })
    
    # Display the overall statistics
    st.header("Population Summary")
    
    # Add interpretation about vector population dynamics
    intrinsic_growth_rate = np.log(total_population[-1] / initial_population) / num_days if total_population[-1] > 0 else 0
    vector_density = total_population[-1] / 100  # Assuming area of 100 square units
    
    # Use calculations based on equations in the chapter
    st.info(f"""
    **Population Growth Analysis:**
    
    According to Black & Moore, the estimated intrinsic growth rate (r) of this vector population is {intrinsic_growth_rate:.4f}, 
    representing the per capita rate of increase.
    
    In the exponential growth model (Fig 24.4), r = {intrinsic_growth_rate:.4f} would produce exponential growth when
    positive or population decline when negative. In reality, as shown in Fig 24.5, logistic growth with carrying capacity
    constraints would eventually limit this growth.
    
    The vector-to-host ratio (m) of {vector_density:.2f} is a critical component of vectorial capacity as shown in 
    MacDonald's equation (Table 24.1).
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Number - Eggs", f"{eggs[-1]:.0f}")
        
    with col2:
        st.metric("Final Number - Immatures", f"{immatures[-1]:.0f}")
        
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
    
    # Tab 1: Population Trends
    with tab1:
        st.header("Population Growth Over Time")
        
        # Population trend plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(days, eggs, label='Eggs', color='#f7d060', linewidth=2)
        ax1.plot(days, immatures, label='Immatures', color='#ff6e40', linewidth=2)
        ax1.plot(days, adults, label='Adults', color='#5d5d5d', linewidth=2)
        ax1.plot(days, total_population, label='Total', color='#1e88e5', linewidth=3, linestyle='--')
        
        ax1.set_xlabel('Day', fontsize=12)
        ax1.set_ylabel('Number of Individuals', fontsize=12)
        ax1.set_title('Vector Population Growth by Life Stage', fontsize=14)
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
            file_name="vector_population_trends.png",
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
            file_name="vector_growth_rate.png",
            mime="image/png"
        )
        
        # Add interpretation text based on chapter
        st.markdown("""
        **Vector Population Growth Dynamics:**
        
        As described by Black & Moore, the population curve shows how vector populations change over time:
        
        1. **Exponential vs. Logistic Growth**: This model demonstrates either exponential growth (Fig 24.4) or 
           oscillating growth patterns similar to the Reed-Frost model (Fig 24.3), depending on parameter settings.
           
        2. **Pulse Reproduction Pattern**: Many vectors like mosquitoes show gonotrophic cycles with distinct 
           pulses of reproduction as females produce batches of eggs at regular intervals (Fig 24.7A).
        
        3. **Implications for Vectorial Capacity**: As shown in Table 24.1, vector density (m) is linearly 
           related to vectorial capacity, while daily survivorship (p) has non-linear effects.
        """)
        
        # Interpretation for growth rate based on chapter
        mean_growth = np.mean(growth_rates[1:])
        st.markdown(f"""
        **Growth Rate Analysis:**
        
        The growth rate plot reveals how vector populations expand or contract over time. This relates to equation (14)
        in the chapter: $N_{{t+1}} = N_t + rN_t = N_t + N_t(b_0 - d_0)$ where:
        
        - Mean daily growth rate: {mean_growth:.2f}%
        - Environments with $b_0 > d_0$ lead to population growth (r > 0)
        - Environments with $d_0 > b_0$ lead to population decline (r < 0)
        
        In natural populations, growth patterns may resemble Fig 24.3 or Fig 24.5 with fluctuations due to:
        
        1. **Carrying Capacity**: As described in the chapter, real populations reach carrying capacity (K) with the 
           logistic growth model shown in Fig 24.5.
        
        2. **Density Dependence**: The text discusses how population density affects individual survival and 
           fecundity, creating natural regulation mechanisms.
        """)
    
    # Tab 2: Stage Distribution
    with tab2:
        st.header("Stage Distribution Analysis")
        
        # Create a stacked area chart for stage proportions
        chart_data = pd.DataFrame({
            'Day': days,
            'Eggs': eggs,
            'Immatures': immatures,
            'Adults': adults
        })
        
        # Reshape for Altair
        chart_data_melted = pd.melt(
            chart_data, 
            id_vars=['Day'], 
            value_vars=['Eggs', 'Immatures', 'Adults'],
            var_name='Stage', 
            value_name='Count'
        )
        
        # Create stacked area chart
        chart = alt.Chart(chart_data_melted).mark_area().encode(
            x='Day:Q',
            y=alt.Y('Count:Q', stack='normalize'),
            color=alt.Color('Stage:N', scale=alt.Scale(
                domain=['Eggs', 'Immatures', 'Adults'],
                range=['#f7d060', '#ff6e40', '#5d5d5d']
            ))
        ).properties(
            width=700,
            height=400,
            title='Relative Proportion of Life Stages Over Time'
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Create a pie chart for the final day
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        final_values = [eggs[-1], immatures[-1], adults[-1]]
        labels = ['Eggs', 'Immatures', 'Adults']
        colors = ['#f7d060', '#ff6e40', '#5d5d5d']
        
        # Only include non-zero values in the pie chart
        non_zero_indices = [i for i, val in enumerate(final_values) if val > 0]
        if non_zero_indices:
            wedges, texts, autotexts = ax3.pie(
                [final_values[i] for i in non_zero_indices], 
                labels=[labels[i] for i in non_zero_indices],
                autopct='%1.1f%%', 
                colors=[colors[i] for i in non_zero_indices], 
                startangle=90,
                shadow=True
            )
            for autotext in autotexts:
                autotext.set_fontsize(12)
                autotext.set_weight('bold')
                
            ax3.set_title(f'Population Composition on Day {num_days}', fontsize=14)
            st.pyplot(fig3)
            
            st.download_button(
                label="Download Population Composition Plot",
                data=fig_to_bytes(fig3),
                file_name="vector_population_composition.png",
                mime="image/png"
            )
        else:
            st.warning("No individuals found in the final day to create pie chart.")
        
        # Interpretation text based on chapter
        st.markdown("""
        **Stable Age Distribution (SAD):**
        
        As shown in Figure 24.11 from Black & Moore, vector populations reach a stable age distribution over time:
        
        1. **Life Stage Proportions**: The proportion of individuals in each life stage stabilizes over time, 
           typically with higher proportions in immature stages than adults due to mortality patterns.
        
        2. **Equilibrium Structure**: This stable distribution is determined by stage-specific survival rates and
           the duration of each life stage. The text emphasizes that a population at SAD has a constant proportion
           of each age class.
        
        3. **Vector Control Implications**: According to the chapter, understanding the age structure is critical
           for designing effective control strategies, as it determines what proportion of the population can be 
           targeted by stage-specific control methods.
        """)
        
        if non_zero_indices:
            st.markdown(f"""
            **Final Population Structure Analysis:**
            
            The population structure shown in the pie chart reflects a population composition influenced by:
            
            1. **Stage-specific survival rates**: Eggs ({egg_survival:.2f}), Immatures ({immature_survival:.2f}), Adults ({adult_survival:.2f})
            2. **Stage durations**: Egg development ({egg_stage_duration} days), Immature development ({immature_stage_duration} days)
            3. **Reproductive pattern**: Following the {fecundity_pattern} pattern shown in Figure 24.7 of the text
            
            According to Figure 24.13 in the chapter, small changes in daily survivorship can dramatically affect
            population structure and size. The relative proportion of life stages is particularly important for
            vector-borne disease epidemiology because only certain stages (typically older adults that have passed
            through the extrinsic incubation period) can transmit pathogens.
            """)
    
    # Tab 3: Age Structure
    with tab3:
        st.header("Age Structure Analysis")
        
        # Fixed list of days to show
        # Calculate some reasonable days to show (beginning, 1/3, 2/3, end)
        day_options = [1, max(1, int(num_days/3)), max(1, int(2*num_days/3)), num_days]
        
        # Let user select which day to focus on using radio buttons
        selected_day = st.radio(
            "Select a day to view age structure:",
            day_options,
            format_func=lambda x: f"Day {x}"
        )
        
        # Create age structure plot for the selected day
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        
        # Get age distribution for the selected day
        day_idx = selected_day - 1  # Convert to 0-based index
        age_distribution = results[day_idx, :]
        
        # Create labels for each age class
        age_labels = []
        for age in range(total_stages):
            if age < egg_stage_duration:
                age_labels.append(f"Egg {age+1}")
            elif age < egg_stage_duration + immature_stage_duration:
                age_labels.append(f"Immature {age+1-egg_stage_duration}")
            else:
                age_labels.append(f"Adult {age+1-(egg_stage_duration+immature_stage_duration)}")
        
        # Color bars by stage
        bar_colors = []
        for age in range(total_stages):
            if age < egg_stage_duration:
                bar_colors.append('#f7d060')  # Egg color
            elif age < egg_stage_duration + immature_stage_duration:
                bar_colors.append('#ff6e40')  # Immature color
            else:
                bar_colors.append('#5d5d5d')  # Adult color
        
        # Highlight reproductive days
        for day, _ in fecundity_values.items():
            if day < total_stages:
                bar_colors[day] = '#e74c3c'  # Highlight reproductive days
        
        # Plot horizontal bar chart
        ax4.barh(range(total_stages), age_distribution, color=bar_colors)
        
        # Set y-ticks (limit to avoid overcrowding)
        max_labels = 25
        if total_stages > max_labels:
            step = max(1, total_stages // max_labels)
            y_ticks = range(0, total_stages, step)
            ax4.set_yticks(y_ticks)
            ax4.set_yticklabels([age_labels[i] for i in y_ticks])
        else:
            ax4.set_yticks(range(total_stages))
            ax4.set_yticklabels(age_labels)
        
        ax4.set_xlabel('Number of Individuals')
        ax4.set_title(f'Age Structure on Day {selected_day}')
        
        # Add stage dividers
        ax4.axhline(y=egg_stage_duration - 0.5, color='k', linestyle='--', alpha=0.3)
        ax4.axhline(y=egg_stage_duration + immature_stage_duration - 0.5, color='k', linestyle='--', alpha=0.3)
        
        st.pyplot(fig4)
    
        st.download_button(
            label="Download Age Structure Plot",
            data=fig_to_bytes(fig4),
            file_name=f"vector_age_structure_day_{selected_day}.png",
            mime="image/png"
        )
        
        # Interpretation text based on chapter
        st.markdown("""
        **Age Structure Dynamics:**
        
        As Black & Moore explain in Figure 24.8, the Leslie Matrix tracks individuals through each day-age class:
        
        1. **Matrix Structure**: The Leslie Matrix contains survival probabilities on the sub-diagonal and 
           fecundity values in the first row, exactly as visualized in Figure 24.8 of the chapter.
        
        2. **Reproductive Ages**: The red bars indicate ages when vectors reproduce, reflecting the reproductive
           patterns shown in Figure 24.7 of the text. These patterns vary from multiple gonotrophic cycles in 
           mosquitoes to single-batch reproduction in ticks.
        
        3. **Age-Based Vectorial Capacity**: The text emphasizes that only older adult vectors that have survived 
           through the extrinsic incubation period can transmit pathogens, making age structure particularly
           important in disease epidemiology.
        """)
        
        # Cohort survival analysis section
        st.subheader("Cohort Survival Analysis")
        
        # Use slider for cohort day selection
        cohort_day = st.slider("Select day to start tracking a cohort:", 1, max(1, num_days-20), 1)
        cohort_day_idx = cohort_day - 1
        
        # Track eggs laid on the selected day
        initial_eggs = results[cohort_day_idx, 0]
        
        if initial_eggs > 0:
            cohort_data = []
            for day in range(cohort_day_idx, min(cohort_day_idx + 28, num_days)):
                age = day - cohort_day_idx
                if age < egg_stage_duration:  # Still an egg
                    individuals = results[day, age]
                    cohort_data.append(("Egg", age+1, individuals))
                elif age < egg_stage_duration + immature_stage_duration:  # Now an immature
                    individuals = results[day, age]
                    cohort_data.append(("Immature", age+1-egg_stage_duration, individuals))
                elif age < total_stages:  # Now an adult
                    individuals = results[day, age]
                    cohort_data.append(("Adult", age+1-(egg_stage_duration+immature_stage_duration), individuals))
                    
            # Create DataFrame for the cohort
            cohort_df = pd.DataFrame(cohort_data, columns=["Stage", "Age", "Count"])
            
            # Calculate survival rate relative to initial eggs
            cohort_df["Survival Rate"] = cohort_df["Count"] / initial_eggs * 100
            
            # Plot cohort survival
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            
            # Get unique stages in the cohort data
            stages = cohort_df["Stage"].unique()
            stage_colors = {'Egg': '#f7d060', 'Immature': '#ff6e40', 'Adult': '#5d5d5d'}
            
            for stage in stages:
                stage_data = cohort_df[cohort_df["Stage"] == stage]
                ax5.plot(stage_data.index + cohort_day, stage_data["Survival Rate"], 
                         label=stage, color=stage_colors[stage], linewidth=2, marker='o')
            
            ax5.set_xlabel('Day', fontsize=12)
            ax5.set_ylabel('Survival Rate (%)', fontsize=12)
            ax5.set_title(f'Cohort Survival from Day {cohort_day}', fontsize=14)
            ax5.legend(fontsize=10)
            ax5.grid(True, alpha=0.3)
            
            # Use log scale for y-axis to match survivorship curves
            ax5.set_yscale('log')
            
            # Add stage transition lines
            transition_days = [
                cohort_day + egg_stage_duration - 1, 
                cohort_day + egg_stage_duration + immature_stage_duration - 1
            ]
            for day in transition_days:
                if day < cohort_day + len(cohort_df):
                    ax5.axvline(x=day, color='k', linestyle='--', alpha=0.5)
            
            st.pyplot(fig5)
            
            st.download_button(
                label="Download Cohort Survival Plot",
                data=fig_to_bytes(fig5),
                file_name="vector_cohort_survival.png",
                mime="image/png"
            )
            
            # Interpretation text based on chapter
            # Determine survivorship curve type based on observed pattern
            if egg_survival > 0.8 and immature_survival > 0.8 and adult_survival < 0.5:
                survival_pattern = "Type I"
            elif abs(egg_survival - immature_survival) < 0.1 and abs(immature_survival - adult_survival) < 0.1:
                survival_pattern = "Type II"
            elif egg_survival < 0.5 and immature_survival > 0.5:
                survival_pattern = "Type III"
            else:
                survival_pattern = "mixed"
            
            st.markdown(f"""
            **Survivorship Classification:**
            
            Based on the survival curve pattern, this population follows a **{survival_pattern}** survivorship pattern.
            As described in Figure 24.6 of the text, there are three basic types of survivorship curves:
            
            - **Type I**: Low mortality at early ages, high mortality at advanced ages (humans, some vectors like tsetse flies)
            - **Type II**: Constant mortality rate throughout life (many vector species with short life spans)
            - **Type III**: High initial mortality among offspring, but survivors have good chances of reaching old age (vectors like ticks)
            
            The text explains that Type II curves are common in many vector species, where a constant fraction of offspring
            is removed in each time period by predators, accidents, or other natural sources of mortality.
            """)
            
            # Calculate estimated daily survival rate
            if len(cohort_df) > 1:
                # Use logarithmic regression to estimate daily survival
                days = np.array(range(len(cohort_df)))
                survival = np.array(cohort_df["Survival Rate"])
                survival_positive = np.maximum(survival, 0.001)  # Avoid log(0)
                
                # Fit log model: log(p) = log(m)/d
                log_survival = np.log(survival_positive/100)
                try:
                    # Simple linear regression on log values
                    slope, _ = np.polyfit(days, log_survival, 1)
                    estimated_daily_survival = np.exp(slope)
                    
                    # Daily survival analysis from chapter
                    st.markdown(f"""
                    **Daily Survival Rate Analysis:**
                    
                    The estimated average daily survival probability from this cohort analysis is **{estimated_daily_survival:.4f}**.
                    
                    According to Table 24.1, daily survivorship (p) is one of the most critical parameters affecting
                    vectorial capacity. A 10% change in p has a disproportionate effect (240% change in vectorial capacity),
                    compared to other parameters. Black & Moore note that daily survival rates can be measured using 
                    mark-release-recapture methods or physiological age grading techniques to track cohorts.
                    
                    The survival rates you've chosen for this simulation:
                    - Eggs: {egg_survival:.2f}
                    - Immatures: {immature_survival:.2f}
                    - Adults: {adult_survival:.2f}
                    
                    The text emphasizes that adult survival rate has the strongest effect on vectorial capacity,
                    as only older adults that have survived through the extrinsic incubation period can transmit pathogens.
                    """)
                except:
                    pass
        else:
            st.warning(f"No eggs were laid on day {cohort_day}. Please select a different day.")
    
    # Tab 4: Data Table
    with tab4:
        st.header("Leslie Matrix & Life Table Analysis")
        
        # Create a heatmap of the Leslie matrix
        leslie_matrix = np.zeros((total_stages, total_stages))
        
        # Set survival probabilities
        for i in range(total_stages-1):
            if i < egg_stage_duration:
                leslie_matrix[i+1, i] = egg_survival
            elif i < egg_stage_duration + immature_stage_duration:
                leslie_matrix[i+1, i] = immature_survival
            else:
                leslie_matrix[i+1, i] = adult_survival
        
        # Set fecundity values
        for day, fecundity in fecundity_values.items():
            if day < total_stages:
                leslie_matrix[0, day] = fecundity
        
        # Create Leslie matrix visualization
        fig6, ax6 = plt.subplots(figsize=(10, 8))
        im = ax6.imshow(leslie_matrix, cmap='viridis')
        plt.colorbar(im, ax=ax6, label='Transition Rate')
        
        # Add lines to separate life stages in the matrix
        ax6.axhline(y=egg_stage_duration - 0.5, color='w', linestyle='-')
        ax6.axhline(y=egg_stage_duration + immature_stage_duration - 0.5, color='w', linestyle='-')
        ax6.axvline(x=egg_stage_duration - 0.5, color='w', linestyle='-')
        ax6.axvline(x=egg_stage_duration + immature_stage_duration - 0.5, color='w', linestyle='-')
        
        # Create custom labels
        x_labels = []
        y_labels = []
        for i in range(total_stages):
            if i == 0:
                x_labels.append('Egg 1')
                y_labels.append('Egg 1')
            elif i < egg_stage_duration:
                x_labels.append(f'Egg {i+1}')
                y_labels.append(f'Egg {i+1}')
            elif i == egg_stage_duration:
                x_labels.append('Immature 1')
                y_labels.append('Immature 1')
            elif i < egg_stage_duration + immature_stage_duration:
                x_labels.append(f'Immature {i-egg_stage_duration+1}')
                y_labels.append(f'Immature {i-egg_stage_duration+1}')
            elif i == egg_stage_duration + immature_stage_duration:
                x_labels.append('Adult 1')
                y_labels.append('Adult 1')
            else:
                x_labels.append(f'Adult {i-(egg_stage_duration+immature_stage_duration)+1}')
                y_labels.append(f'Adult {i-(egg_stage_duration+immature_stage_duration)+1}')
        
        # Set reduced ticks to avoid overcrowding
        max_ticks = 20
        stride = max(1, total_stages // max_ticks)
        
        ax6.set_xticks(range(0, total_stages, stride))
        ax6.set_yticks(range(0, total_stages, stride))
        ax6.set_xticklabels([x_labels[i] for i in range(0, total_stages, stride)], rotation=90)
        ax6.set_yticklabels([y_labels[i] for i in range(0, total_stages, stride)])
        
        ax6.set_title('Leslie Matrix Visualization', fontsize=14)
        ax6.set_xlabel('Current Age (From)', fontsize=12)
        ax6.set_ylabel('Next Age (To)', fontsize=12)
        
        st.pyplot(fig6)
        
        st.download_button(
            label="Download Leslie Matrix Plot",
            data=fig_to_bytes(fig6),
            file_name="vector_leslie_matrix.png",
            mime="image/png"
        )
        
        # Interpretation text based on chapter
        st.markdown("""
        **Leslie-Lewis Matrix Model:**
        
        This visualization shows the Leslie Matrix as described in Figure 24.8 of the chapter:
        
        1. **Matrix Structure**: As described by Black & Moore, the Leslie Matrix contains:
           - Fecundity values in the first row (reproductive output at each age)
           - Survival probabilities on the sub-diagonal (transition rates between age classes)
           - Zeros elsewhere
        
        2. **Matrix Multiplication**: The Leslie-Lewis approach projects the current population to future time steps
           through matrix multiplication: $n_{t+1} = L n_t$ where L is the Leslie Matrix and $n_t$ is the
           population vector by age class.
        
        3. **Stable Age Distribution**: An important result of iterating this matrix, as shown in Figure 24.11,
           is that the population reaches a stable age distribution (SAD) where the proportion of each age class
           remains constant, even as total population size continues to grow.
        """)
        
        # Show the detailed data table
        st.subheader("Population Data by Day")
        
        # Life table interpretation from chapter
        st.markdown("""
        **Life Table Analysis:**
        
        Tables 24.2 and 24.3 in the text show examples of life tables for vectors:
        
        1. **Life Table Components**: As described by Black & Moore, life tables combine:
           - Chronological age (days)
           - Age-specific survival rates (p)
           - Age-specific fecundity (f)
        
        2. **Age Classes**: The text emphasizes that "age classes must be of equal length for the model to be accurate."
           This model uses day-age classes rather than life stages to ensure precise tracking of individuals.
        
        3. **Applications**: Life tables enable the prediction of population growth and age structure,
           which are critical for understanding vectorial capacity and the efficacy of control measures.
        """)
        
        # Allow users to choose data resolution
        data_resolution = st.radio(
            "Data Resolution:",
            ["Summary (by life stage)", "Detailed (by day and age)"]
        )
        
        if data_resolution == "Summary (by life stage)":
            st.dataframe(summary_df.style.background_gradient(cmap='viridis', subset=['Total']))
            
            # Download button for CSV
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary Data as CSV",
                data=csv,
                file_name="vector_leslie_matrix_summary.csv",
                mime="text/csv"
            )
        else:
            # Create a detailed dataframe with day-by-age data
            detailed_columns = []
            for i in range(total_stages):
                if i < egg_stage_duration:
                    detailed_columns.append(f'Egg {i+1}')
                elif i < egg_stage_duration + immature_stage_duration:
                    detailed_columns.append(f'Immature {i-egg_stage_duration+1}')
                else:
                    detailed_columns.append(f'Adult {i-(egg_stage_duration+immature_stage_duration)+1}')
            
            detailed_df = pd.DataFrame(results, columns=detailed_columns)
            detailed_df.insert(0, 'Day', days)
            detailed_df['Eggs'] = eggs
            detailed_df['Immatures'] = immatures
            detailed_df['Adults'] = adults
            detailed_df['Total'] = total_population
            
            st.dataframe(detailed_df.style.background_gradient(cmap='viridis', subset=['Total']))
            
            # Download button for detailed CSV
            detailed_csv = detailed_df.to_csv(index=False)
            st.download_button(
                label="Download Detailed Data as CSV",
                data=detailed_csv,
                file_name="vector_leslie_matrix_detailed.csv",
                mime="text/csv"
            )
    
    # Add section on management implications
    st.header("Implications for Vector Control")
    
    # Create comparative plot for control strategies
    # Create data for three scenarios - baseline, immature control, adult control
    control_days = np.arange(1, 101)
    
    # Scenario 1: Baseline (current parameters)
    baseline_adults = adults[:min(100, len(adults))]
    if len(baseline_adults) < 100:
        baseline_adults = np.pad(baseline_adults, (0, 100-len(baseline_adults)), 'constant', constant_values=baseline_adults[-1] if len(baseline_adults) > 0 else 0)
    
    # Scenario 2: Simulated immature control (50% reduction in immature survival)
    immature_control_results, _, _, _ = run_leslie_model(
        egg_survival, immature_survival * 0.5, adult_survival, 
        initial_population, fecundity_values, 
        min(100, num_days), egg_stage_duration, immature_stage_duration
    )
    immature_control_adults = np.sum(immature_control_results[:, adult_indices], axis=1)
    if len(immature_control_adults) < 100:
        immature_control_adults = np.pad(immature_control_adults, (0, 100-len(immature_control_adults)), 'constant', constant_values=immature_control_adults[-1] if len(immature_control_adults) > 0 else 0)
    
    # Scenario 3: Simulated adult control (50% reduction in adult survival)
    adult_control_results, _, _, _ = run_leslie_model(
        egg_survival, immature_survival, adult_survival * 0.5, 
        initial_population, fecundity_values, 
        min(100, num_days), egg_stage_duration, immature_stage_duration
    )
    adult_control_adults = np.sum(adult_control_results[:, adult_indices], axis=1)
    if len(adult_control_adults) < 100:
        adult_control_adults = np.pad(adult_control_adults, (0, 100-len(adult_control_adults)), 'constant', constant_values=adult_control_adults[-1] if len(adult_control_adults) > 0 else 0)
    
    # Plot the comparison
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    ax7.plot(control_days, baseline_adults, label='Baseline', color='#1e88e5', linewidth=2)
    ax7.plot(control_days, immature_control_adults, label='50% Reduction in Immature Survival', color='#ff6e40', linewidth=2)
    ax7.plot(control_days, adult_control_adults, label='50% Reduction in Adult Survival', color='#5d5d5d', linewidth=2)
    
    ax7.set_xlabel('Day', fontsize=12)
    ax7.set_ylabel('Number of Adult Vectors', fontsize=12)
    ax7.set_title('Effect of Control Strategies on Adult Vector Population', fontsize=14)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # Use log scale if numbers get large
    if max(baseline_adults) > 1000 or max(immature_control_adults) > 1000 or max(adult_control_adults) > 1000:
        ax7.set_yscale('log')
        st.info("Using logarithmic scale for y-axis due to large population numbers")
    
    st.pyplot(fig7)
    
    # Control strategies interpretation from chapter
    st.markdown("""
    ### Vector Control Strategy Insights
    
    Black & Moore discuss critical implications for vector control:
    
    1. **Vectorial Capacity Targeting**:
       - As shown in Table 24.1, vectorial capacity (V) is affected differently by various parameters
       - Small changes in daily survival rate (p) have substantial impacts on vectorial capacity
       - Figure 24.13 demonstrates how reducing survival rates affects final population size
    
    2. **Economic Thresholds**:
       - The text discusses that vector control aims to keep populations below economic thresholds
       - Control efforts should focus on the most cost-effective parameters to target
       - Mathematical models help identify optimal control strategies
    
    3. **Larval vs Adult Control**:
       - The authors note that targeting different life stages produces different outcomes
       - Figure 24.13 shows that reducing egg survivorship may be more efficient than targeting adults
       - The text cautions that density dependence can complicate control outcomes - reducing larval 
         numbers can sometimes lead to better survival of remaining larvae
    """)
    
    # Add conclusions
    st.header("Connecting Population Biology to Vector Management")
    
    # Conclusion based on chapter
    st.markdown("""
    ### Practical Applications of Population Models
    
    In their conclusion, Black & Moore emphasize several important points:
    
    1. **Model Limitations**:
       - The authors explicitly caution: "these models are only didactic tools and 
         are largely inaccurate in their predictions concerning natural populations"
       - Real vector populations are more complex than these idealized representations
       - Models should be used for comparative rather than absolute predictions
    
    2. **Data Collection Challenges**:
       - The text describes methods for measuring key parameters like survival rates and fecundity
       - Techniques include mark-release-recapture, physiological age grading, and cohort analysis
       - Accurate field measurements are challenging but essential
    
    3. **Comparative Approach**:
       - Black & Moore recommend using a "comparative" rather than "absolute" approach
       - Understanding the relative importance of different factors is more useful than obtaining precise values
       - These models can guide control strategies by identifying which parameters most strongly influence vectorial capacity
    
    These mathematical approaches demonstrate how population biology can guide practical vector control
    by identifying the most effective intervention points based on understanding population structure and dynamics.
    """)

if __name__ == "__main__":
    run()
