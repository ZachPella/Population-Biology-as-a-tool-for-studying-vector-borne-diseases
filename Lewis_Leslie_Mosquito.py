import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io

def run():    
    # Display title and description with academic context
    st.title("🦟 Leslie Matrix Mosquito Population Model")
    st.markdown("#### 🧬 A discrete, age-structured population dynamics simulator")
    st.markdown("""
    This interactive application simulates mosquito population dynamics using a Leslie Matrix model based on 
    population biology principles for studying vector and pest populations.
    Adjust the parameters using the sliders and see how they affect the population growth.
    
    **Definition**: The Leslie Matrix model is a discrete-time, age-structured population model that describes 
    population growth when age-specific survival rates and reproductive rates can be estimated. It's a fundamental 
    mathematical tool in population ecology, representing stage transitions and fecundity in a matrix format.
    
    **Core Concept**: At the heart of the Leslie Matrix model is the projection of current population structure 
    to future time steps through matrix multiplication. The matrix contains survival probabilities on the sub-diagonal 
    and fecundity values in the first row.
    
    **Leslie Matrix Equation**: 
    $n_{t+1} = M × n_{t}$
    
    Where:
    - $n_{t+1}$ is the population vector at the next time step
    - $M$ is the Leslie Matrix containing survival rates and fecundity values
    - $n_{t+1}$ is the current population vector by age class
    
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
    
    
    Where:
    - $m$ = vector density in relation to hosts
    - $a$ = biting rate (humans bitten per mosquito per day)
    - $p$ = daily survival rate (especially of adults)
    - $n$ = extrinsic incubation period (days from ingestion to transmission)
    
    ### Adjustable Parameters:
    - **Survival rates** for each life stage (eggs, larvae, adults)
    - **Duration** of each life stage
    - **Fecundity** after each blood meal
    - **Initial population** size
    - **Density dependence** and immigration (optional)
    
    This model demonstrates how small changes in daily survival rates and age structure can dramatically affect both population growth and disease transmission potential.
    """)
    
    # Create sidebar with ALL parameters
    st.sidebar.header("Model Parameters")
    
    # Life Stage Durations in sidebar
    st.sidebar.subheader("Life Stage Durations")
    egg_stage_duration = st.sidebar.slider("Egg stage duration (days)", 1, 10, 2, 
                                  help="Duration of egg development before hatching")
    larval_stage_duration = st.sidebar.slider("Larval stage duration (days)", 1, 20, 10, 
                                      help="Duration of larval and pupal stages before emerging as adults")
    
    # Survival Rates in sidebar
    st.sidebar.subheader("Survival Rates")
    egg_survival = st.sidebar.slider("Egg daily survival rate", 0.0, 1.0, 0.5, 0.01, 
                              help="Daily probability an egg survives (Table 24.2: typically 0.4-0.6)")
    larval_survival = st.sidebar.slider("Larval daily survival rate", 0.0, 1.0, 0.6, 0.01,
                                 help="Daily probability a larva survives")
    adult_survival = st.sidebar.slider("Adult daily survival rate", 0.0, 1.0, 0.9, 0.01, 
                                help="Daily probability an adult survives (typically 0.8-0.95)")
    
    # Population Parameters in sidebar
    st.sidebar.subheader("Population Parameters")
    initial_population = st.sidebar.number_input("Initial population (adults)", 1, 10000, 100, 
                                         help="Starting population of adult mosquitoes")
    num_days = st.sidebar.slider("Simulation length (days)", 30, 365, 100, 
                         help="Length of simulation in days")
    
    # Fecundity Values in sidebar
    st.sidebar.subheader("Fecundity Values")
    fecundity_1 = st.sidebar.number_input("Fecundity after first blood meal (day 12)", 0, 500, 120, 
                                   help="Eggs produced after first blood meal")
    fecundity_2 = st.sidebar.number_input("Fecundity after second blood meal (day 17)", 0, 500, 100, 
                                   help="Eggs produced after second blood meal")
    fecundity_3 = st.sidebar.number_input("Fecundity after third blood meal (day 22)", 0, 500, 80, 
                                   help="Eggs produced after third blood meal")
    fecundity_4 = st.sidebar.number_input("Fecundity after fourth blood meal (day 27)", 0, 500, 60, 
                                   help="Eggs produced after fourth blood meal")
    
    # Advanced features in sidebar
    st.sidebar.subheader("Advanced Model Features")
    enable_density_dependence = st.sidebar.checkbox("Enable Density Dependence", False, 
                                            help="Implement logistic growth (Fig 24.5)")
    if enable_density_dependence:
        carrying_capacity = st.sidebar.number_input("Carrying Capacity (K)", 100, 1000000, 50000, 
                                           help="Maximum sustainable population size")
    else:
        carrying_capacity = 50000
        
    enable_immigration = st.sidebar.checkbox("Enable Immigration & Mortality", False, 
                                     help="Allow immigration/emigration (Fig 24.3)")
    if enable_immigration:
        immigration_rate = st.sidebar.slider("Immigration rate (adults per 2 days)", 0, 100, 10)
        mortality_rate = st.sidebar.slider("Mortality rate (fraction leaving)", 0.0, 1.0, 0.1)
    else:
        immigration_rate = 10
        mortality_rate = 0.1

    # Model code
    def run_leslie_model(egg_survival, larval_survival, adult_survival, 
                        initial_pop, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
                        num_days, egg_stage_duration, larval_stage_duration,
                        enable_density_dependence=False, carrying_capacity=50000,
                        enable_immigration=False, immigration_rate=10, mortality_rate=0.1):
        """
        Run the Leslie Matrix population model simulation for mosquitoes
        """
        adult_stage_start = egg_stage_duration + larval_stage_duration
        total_stages = max(28, adult_stage_start + 20)  # Ensure we have enough stages
        
        # Create the Leslie Matrix (M matrix in Fig 24.8)
        leslie_matrix = np.zeros((total_stages, total_stages))
        
        # Set survival probabilities (subdiagonal) - P values in Table 24.2
        for i in range(total_stages-1):
            if i < egg_stage_duration:  # Egg stage
                leslie_matrix[i+1, i] = egg_survival
            elif i < egg_stage_duration + larval_stage_duration:  # Larval stage
                leslie_matrix[i+1, i] = larval_survival
            else:  # Adult stage
                leslie_matrix[i+1, i] = adult_survival
        
        # Set fecundity values (first row) - F values in Table 24.2
        # Reproduction occurs on specific adult days (after blood meals)
        reproduction_days = [12, 17, 22, 27]  # Blood meal days
        
        # Ensure indices are valid
        for i, day in enumerate(reproduction_days):
            if adult_stage_start + (day - adult_stage_start) < total_stages:
                if day >= adult_stage_start:
                    leslie_matrix[0, day] = [fecundity_1, fecundity_2, fecundity_3, fecundity_4][i]
        
        # Initialize population vector (n vector in Fig 24.8)
        population = np.zeros(total_stages)
        population[adult_stage_start] = initial_pop  # Start with initial adults
        
        # Initialize results matrix
        results = np.zeros((num_days, total_stages))
        results[0, :] = population
        
        # Run the simulation - similar to iteration process described in Fig 24.9
        for day in range(1, num_days):
            # Apply Leslie matrix to get next day's population without adjustments
            new_population = leslie_matrix @ population
            
            # Apply density dependence if enabled (based on Fig 24.5 logistic growth)
            if enable_density_dependence:
                total_population = np.sum(population)
                if total_population > 0:
                    # Apply logistic scaling factor based on equation 15 in the text
                    scaling_factor = 1 + ((carrying_capacity - total_population) / carrying_capacity)
                    scaling_factor = max(0.1, min(scaling_factor, 1.5))  # Limit scaling to prevent extremes
                    new_population = new_population * scaling_factor
            
            # Apply immigration and mortality if enabled (based on Fig 24.3)
            if enable_immigration and day % 2 == 0:  # Every 2 days
                # Add new adults from immigration
                new_population[adult_stage_start] += immigration_rate
                
                # Apply mortality to cases - removing a fraction of all individuals
                adult_indices = range(egg_stage_duration + larval_stage_duration, total_stages)
                adult_count = np.sum(new_population[adult_indices])
                
                # Calculate how many adults die
                adults_dying = int(adult_count * mortality_rate / 10)  # Scale down to make effect visible
                
                # Remove adults, starting from oldest
                if adults_dying > 0 and adult_count > 0:
                    for i in range(total_stages-1, egg_stage_duration + larval_stage_duration-1, -1):
                        if adults_dying <= 0:
                            break
                        dying_from_this_age = min(adults_dying, new_population[i])
                        new_population[i] -= dying_from_this_age
                        adults_dying -= dying_from_this_age
            
            population = new_population
            results[day, :] = population
        
        return results, total_stages, egg_stage_duration, larval_stage_duration
    
    # Run the model
    results, total_stages, egg_stage_duration, larval_stage_duration = run_leslie_model(
        egg_survival, larval_survival, adult_survival, 
        initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
        num_days, egg_stage_duration, larval_stage_duration,
        enable_density_dependence, carrying_capacity,
        enable_immigration, immigration_rate, mortality_rate
    )
    
    # Process results for visualization
    days = np.arange(1, num_days+1)
    egg_indices = range(0, egg_stage_duration)
    larva_indices = range(egg_stage_duration, egg_stage_duration + larval_stage_duration)
    adult_indices = range(egg_stage_duration + larval_stage_duration, total_stages)
    
    # Calculate life stage totals
    eggs = np.sum(results[:, egg_indices], axis=1)
    larvae = np.sum(results[:, larva_indices], axis=1)
    adults = np.sum(results[:, adult_indices], axis=1)
    total_population = eggs + larvae + adults
    
    # Calculate stage percentages for stable age distribution analysis (Fig 24.11)
    percent_eggs = np.zeros(num_days)
    percent_larvae = np.zeros(num_days)
    percent_adults = np.zeros(num_days)
    
    for i in range(num_days):
        if total_population[i] > 0:
            percent_eggs[i] = (eggs[i] / total_population[i]) * 100
            percent_larvae[i] = (larvae[i] / total_population[i]) * 100
            percent_adults[i] = (adults[i] / total_population[i]) * 100
    
    # Calculate vectorial capacity
    extrinsic_incubation = 10  # Days from ingestion to transmission capability
    biting_rate = 0.25  # Bites per vector per day
    vector_competence = 0.5  # Proportion of bites that successfully infect
    
    vectorial_capacity = adults[-1] * (adult_survival ** extrinsic_incubation) * biting_rate**2 * vector_competence / (-np.log(adult_survival))
    
    # Calculate key metrics
    peak_adults = np.max(adults)
    peak_time = np.argmax(adults)
    growth_rate = 100 * (np.exp(np.log(total_population[-1]/initial_population)/num_days) - 1) if total_population[-1] > 0 and initial_population > 0 else 0
    
    # Summary Statistics styled like Reed-Frost
    st.header("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Peak Adult Population", f"{int(peak_adults):,}")
    with col2:
        st.metric("Time to Peak", f"{peak_time} days")
    with col3:
        st.metric("Final Adult Population", f"{int(adults[-1]):,}")
    with col4:
        st.metric("Vectorial Capacity", f"{vectorial_capacity:.2f}")

    # Helper function to convert figure to downloadable data
    def fig_to_bytes(fig):
        """Convert a matplotlib figure to bytes for downloading"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        return buf
    
    # Create tabs for different views - match the cockroach model tab structure
    tab1, tab2, tab3, tab4 = st.tabs([
        "Population Trends", 
        "Stage Distribution", 
        "Age Structure", 
        "Data Table"
    ])
    
    with tab1:
        st.header("Population Growth Over Time")
        
        # Population trend plot FIRST
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(days, eggs, label='Eggs', color='#f7d060', linewidth=2)
        ax1.plot(days, larvae, label='Larvae', color='#ff6e40', linewidth=2)
        ax1.plot(days, adults, label='Adults', color='#5d5d5d', linewidth=2)
        ax1.plot(days, total_population, label='Total', color='#1e88e5', linewidth=3, linestyle='--')
        
        ax1.set_xlabel('Day', fontsize=12)
        ax1.set_ylabel('Number of Individuals', fontsize=12)
        ax1.set_title('Mosquito Population Growth by Life Stage', fontsize=14)
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
            file_name="mosquito_population_trends.png",
            mime="image/png"
        )
    
        # INTERPRETATION AFTER - match cockroach model style
        st.subheader("Population Dynamics Interpretation")
    
        st.markdown("""
        **Mosquito Population Growth Dynamics:**
        
        This visualization reveals characteristic growth patterns in mosquito populations, with several important features:
        
        1. **Pulse Reproduction Pattern**: Mosquitoes show distinct pulses of reproduction as females take blood meals at 
           regular intervals, creating the step-like increases in egg numbers. This pattern corresponds directly to the 
           gonotrophic cycles described in Black & Moore.
        
        2. **Stage Duration Effects**: The relative lengths of egg and larval stages create a time-delayed pattern of 
           population growth, with waves of individuals moving through developmental stages. This developmental timing 
           directly influences vectorial capacity.
        
        3. **Control Implications**: For vector management, this growth curve suggests that interventions targeting 
           adults before they reproduce will be most effective at reducing disease transmission. The exponential nature 
           of the growth curve demonstrates why early detection and intervention are crucial.
        """)
        
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
        
        # Growth rate interpretation - match cockroach model style
        mean_growth = np.mean(growth_rates[1:])
        st.pyplot(fig2)
        
        st.markdown(f"""
        **Growth Rate Pattern Analysis:**
        
        The growth rate plot reveals how mosquito populations expand over time. The mean daily growth rate 
        of {mean_growth:.2f}% translates to significant population increases over short periods.
        
        Key observations:
        
        1. **Cyclical Growth Patterns**: The spikes in growth rate correspond to the maturation of large 
           cohorts of larvae into reproductive adults, followed by blood meals and egg-laying events. These 
           cycles create predictable windows for control interventions.
        
        2. **Initial Establishment Phase**: Vector populations typically show a lag phase with lower 
           growth rates followed by a rapid acceleration as multiple reproducing females become established. 
           This pattern mirrors the logistic growth equation described in Black & Moore's chapter.
        
        3. **Carrying Capacity Effects**: When density dependence is enabled, growth rates would eventually decline 
           as populations approach carrying capacity due to resource limitations and density-dependent factors.
        """)
    
    with tab2:
        st.header("Stage Distribution Analysis")
        
        # Create a stacked area chart for stage proportions
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
            title='Relative Proportion of Life Stages Over Time (Stable Age Distribution)'
        )
        
        st.altair_chart(chart, use_container_width=True)
        
        # Create a pie chart for the final day
        fig3, ax3 = plt.subplots(figsize=(8, 8))
        final_values = [eggs[-1], larvae[-1], adults[-1]]
        labels = ['Eggs', 'Larvae', 'Adults']
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
            
            # INTERPRETATION - match cockroach model style
            st.subheader("Stage Distribution Interpretation")
            st.markdown("""
            **Mosquito Stage Distribution Dynamics:**
            
            This visualization shows how the mosquito population structure evolves over time, eventually reaching 
            a stable age distribution (SAD) - a fundamental concept in population biology described in Black & Moore. 
            In mosquito populations, this structural pattern has important implications:
            
            1. **Hidden Reservoir Effect**: The high proportion of eggs and larvae (often >70% of the total 
               population) explains why mosquito populations can quickly rebound after control efforts targeting only adults.
               This "population iceberg" effect is critical for vector management planning.
            
            2. **Resilience Mechanism**: The extended development time of larvae creates a buffer against control 
               measures - even if all adults are eliminated, the larval reservoir ensures population recovery. 
               This resilience mechanism is similar to the "demographic momentum" described in population models.
            
            3. **Vectorial Capacity Implications**: The stable age distribution directly affects disease transmission potential
               by determining what proportion of the population survives to ages capable of transmitting pathogens.
               As described in Macdonald's equation, only older females contribute to disease spread.
            """)
            
            st.download_button(
                label="Download Population Composition Plot",
                data=fig_to_bytes(fig3),
                file_name="mosquito_population_composition.png",
                mime="image/png"
            )
        else:
            st.warning("No individuals found in the final day to create pie chart.")
    
    with tab3:
        st.header("Age Structure Analysis")
        
        # Let user select which day to focus on using radio buttons
        day_options = [1, max(1, int(num_days/3)), max(1, int(2*num_days/3)), num_days]
        selected_day = st.radio(
            "Select a day to view age structure:",
            day_options,
            format_func=lambda x: f"Day {x}",
            horizontal=True
        )
        
        # Create age structure plot for the selected day
        day_idx = selected_day - 1  # Convert to 0-based index
        age_distribution = results[day_idx, :]
        
        # Create labels for each age class
        age_labels = []
        for age in range(total_stages):
            if age < egg_stage_duration:
                age_labels.append(f"Egg {age+1}")
            elif age < egg_stage_duration + larval_stage_duration:
                age_labels.append(f"Larva {age+1-egg_stage_duration}")
            else:
                age_labels.append(f"Adult {age+1-(egg_stage_duration+larval_stage_duration)}")
        
        # Color bars by stage
        bar_colors = []
        for age in range(total_stages):
            if age < egg_stage_duration:
                bar_colors.append('#f7d060')  # Egg color
            elif age < egg_stage_duration + larval_stage_duration:
                bar_colors.append('#ff6e40')  # Larva color
            else:
                bar_colors.append('#5d5d5d')  # Adult color
        
        adult_stage_start = egg_stage_duration + larval_stage_duration
        
        # Highlight reproduction days
        for age in range(total_stages):
            if age in [12, 17, 22, 27]:  # Blood meal days
                bar_colors[age] = '#e74c3c'  # Highlight reproduction days
        
        # Plot horizontal bar chart
        fig4, ax4 = plt.subplots(figsize=(10, 8))
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
        ax4.set_title(f'Age Structure on Day {selected_day}', fontsize=14)
        
        # Add stage dividers
        ax4.axhline(y=egg_stage_duration - 0.5, color='k', linestyle='--', alpha=0.3)
        ax4.axhline(y=egg_stage_duration + larval_stage_duration - 0.5, color='k', linestyle='--', alpha=0.3)
        
        st.pyplot(fig4)
        
        st.download_button(
            label="Download Age Structure Plot",
            data=fig_to_bytes(fig4),
            file_name=f"mosquito_age_structure_day_{selected_day}.png",
            mime="image/png"
        )
        
        # INTERPRETATION - match cockroach model style
        st.subheader("Age Structure Interpretation")
        
        st.markdown("""
        **Mosquito Population Age Structure:**
        
        This visualization displays the detailed age distribution of the mosquito population, revealing patterns 
        that are not apparent in the aggregated life stage counts:
        
        1. **Blood Meal Timing**: The red bars indicate ages when females take blood meals and produce eggs. 
           Unlike species with continuous reproduction, mosquitoes show distinct reproductive pulses following 
           each blood meal, creating cohorts that move through the population together.
        
        2. **Development Bottlenecks**: Transitions between life stages (marked by dotted lines) represent 
           vulnerable periods in the mosquito life cycle. For example, pupation and adult emergence may be 
           more susceptible to environmental stressors, creating windows of opportunity for control.
        
        3. **Transmission Implications**: Only adult mosquitoes that survive beyond the extrinsic incubation 
           period (typically 10-14 days post-emergence) can transmit disease. This age structure directly 
           affects vectorial capacity as described in Black & Moore's model.
        
        4. **Cohort Identification**: The distinct "waves" of individuals at specific ages indicates separate 
           cohorts moving through the population, reflecting the pulsed reproduction pattern following gonotrophic cycles.
        """)
        
        # Cohort survival analysis
        st.subheader("Cohort Survival Analysis")
        
        # Use simpler slider for cohort day selection
        cohort_day = st.slider("Select day to start tracking a cohort:", 1, max(1, num_days-10), 1)
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
                elif age < egg_stage_duration + larval_stage_duration:  # Now a larva
                    individuals = results[day, age]
                    cohort_data.append(("Larva", age+1-egg_stage_duration, individuals))
                elif age < total_stages:  # Now an adult
                    individuals = results[day, age]
                    cohort_data.append(("Adult", age+1-(egg_stage_duration+larval_stage_duration), individuals))
                    
            # Create DataFrame for the cohort
            cohort_df = pd.DataFrame(cohort_data, columns=["Stage", "Age", "Count"])
            
            # Calculate survival rate relative to initial eggs
            cohort_df["Survival Rate"] = cohort_df["Count"] / initial_eggs * 100
            
            # Plot cohort survival
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            
            # Get unique stages in the cohort data
            stages = cohort_df["Stage"].unique()
            stage_colors = {'Egg': '#f7d060', 'Larva': '#ff6e40', 'Adult': '#5d5d5d'}
            
            for stage in stages:
                stage_data = cohort_df[cohort_df["Stage"] == stage]
                ax5.plot(stage_data.index + cohort_day, stage_data["Survival Rate"], 
                         label=stage, color=stage_colors[stage], linewidth=2, marker='o')
            
            ax5.set_xlabel('Day', fontsize=12)
            ax5.set_ylabel('Survival Rate (%)', fontsize=12)
            ax5.set_title(f'Cohort Survival from Day {cohort_day}', fontsize=14)
            ax5.legend(fontsize=10)
            ax5.grid(True, alpha=0.3)
            
            #
