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
    representing the per capita rate of increase as described in the exponential growth model equation (14) in Black & Moore.
    
    For pest species like cockroaches, population models help identify critical life stages and 
    thresholds for effective control measures. The growth potential shown here demonstrates the 
    importance of early intervention in pest management.
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
        
        # UNIQUE INTERPRETATION FOR POPULATION TRENDS TAB - BASED ON BLACK & MOORE CHAPTER
        st.markdown("""
        **Cockroach Population Growth Dynamics:**
        
        This visualization reveals characteristic growth patterns in cockroach populations as discussed by Black & Moore:
        
        1. **Pulse Reproduction Pattern**: Similar to the pattern shown in Figure 24.7A, cockroaches show distinct pulses of 
           reproduction as females produce oothecae at regular intervals, creating the step-like increases in 
           egg numbers. This pattern resembles the gonotrophic cycles described for some vector species.
        
        2. **Stage Duration Effects**: The relative lengths of egg and nymphal stages create a time-delayed 
           pattern of population growth, with waves of individuals moving through developmental stages. The 
           longer the nymphal stage, the greater the delay between reproduction and the emergence of new 
           reproductive adults.
        
        3. **Control Implications**: The exponential or logistic pattern displayed in this model (similar to Fig. 24.4 and 24.5)
           suggests that interventions targeting adults before they reproduce would be most effective at preventing 
           population explosions. The growth curve demonstrates why early detection and intervention are crucial.
        """)
        
        # UNIQUE INTERPRETATION FOR GROWTH RATE - BASED ON BLACK & MOORE CHAPTER
        mean_growth = np.mean(growth_rates[1:])
        st.markdown(f"""
        **Growth Rate Pattern Analysis:**
        
        The growth rate plot reveals how cockroach populations expand over time, with a mean daily growth rate 
        of {mean_growth:.2f}%. This relates to the equation Nt+1 = Nt + rNt discussed by Black & Moore.
        
        Key observations:
        
        1. **Cyclical Growth Patterns**: The spikes in growth rate correspond to the maturation of large 
           cohorts of nymphs into reproductive adults, followed by the production of new egg cases. In natural
           settings, these cycles might create predictable windows for control interventions.
        
        2. **Initial Establishment Phase**: Population growth shows an initial phase followed by acceleration,
           similar to patterns described in the text. The pattern relates to equation (14) in the chapter where 
           b₀ > d₀ results in positive population growth.
        
        3. **Carrying Capacity Effects**: In natural settings, growth rates would eventually decline as 
           populations approach carrying capacity (as shown in Fig. 24.5) due to resource limitations and 
           density-dependent factors. In human habitations, this ceiling may be much higher than in natural environments.
        """)
    
    with tab2:
        st.header("Stage Distribution Analysis")
        
        # Create a stacked area chart for stage proportions
        chart_data = pd.DataFrame({
            'Day': days,
            'Eggs': eggs,
            'Nymphs': nymphs,
            'Adults': adults
        })
        
        # Reshape for Altair
        chart_data_melted = pd.melt(
            chart_data, 
            id_vars=['Day'], 
            value_vars=['Eggs', 'Nymphs', 'Adults'],
            var_name='Stage', 
            value_name='Count'
        )
        
        # Create stacked area chart
        chart = alt.Chart(chart_data_melted).mark_area().encode(
            x='Day:Q',
            y=alt.Y('Count:Q', stack='normalize'),
            color=alt.Color('Stage:N', scale=alt.Scale(
                domain=['Eggs', 'Nymphs', 'Adults'],
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
        final_values = [eggs[-1], nymphs[-1], adults[-1]]
        labels = ['Eggs', 'Nymphs', 'Adults']
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
                file_name="cockroach_population_composition.png",
                mime="image/png"
            )
        else:
            st.warning("No individuals found in the final day to create pie chart.")
        
        # UNIQUE INTERPRETATION FOR STAGE DISTRIBUTION - BASED ON BLACK & MOORE CHAPTER
        st.markdown("""
        **Cockroach Stage Distribution Dynamics:**
        
        This visualization shows how the cockroach population structure evolves over time, eventually reaching 
        a stable age distribution (SAD) as described in Figure 24.11 of Black & Moore. In cockroach populations, 
        this structural pattern has important implications:
        
        1. **Hidden Infestation Indicators**: The high proportion of eggs and nymphs (often >70% of the total 
           population) explains why cockroach infestations are often much larger than apparent from visible adults. 
           This "population iceberg" effect is critical for pest management planning.
        
        2. **Resilience Mechanism**: The extended development time of nymphs creates a buffer against control 
           measures - even if all adults are eliminated, the nymphal reservoir ensures population recovery. 
           This resilience mechanism is similar to the demographic dynamics described in the text.
        
        3. **Control Strategy Implications**: The eventual stable proportion of each life stage determines what 
           fraction of the population can be targeted by stage-specific control methods (e.g., growth regulators 
           vs. adult baits). Effective management requires addressing all life stages present in the stable 
           distribution.
        """)
        
        if non_zero_indices:
            # UNIQUE INTERPRETATION FOR FINAL PROPORTIONS - BASED ON BLACK & MOORE CHAPTER
            st.markdown(f"""
            **Final Population Structure Analysis:**
            
            The pie chart shows the stable age distribution reached after {num_days} days. According to the concepts in
            Black & Moore, this equilibrium structure is determined by:
            
            1. **Stage-specific survival rates**: Eggs ({egg_survival:.2f}), Nymphs ({nymphal_survival:.2f}), Adults ({adult_survival:.2f})
            2. **Stage durations**: Egg development ({egg_stage_duration} days), Nymphal development ({nymphal_stage_duration} days)
            3. **Reproductive pattern**: Similar to Fig 24.7A, with oviposition occurring at days 12, 17, 22, and 27 of adult life
            
            Cockroach populations typically show a much higher proportion of nymphs than adults due to their 
            extended developmental period. This creates challenges for control programs, as the majority of the 
            population is in life stages that may be less susceptible to certain control measures and often 
            hide in inaccessible locations.
            
            As noted in the text, this stable structure helps explain why cockroach control often requires sustained effort 
            rather than one-time treatments - the age distribution ensures a constant supply of developing 
            individuals to replace eliminated adults.
            """)
    
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
            elif age < egg_stage_duration + nymphal_stage_duration:
                age_labels.append(f"Nymph {age+1-egg_stage_duration}")
            else:
                age_labels.append(f"Adult {age+1-(egg_stage_duration+nymphal_stage_duration)}")
        
        # Color bars by stage
        bar_colors = []
        for age in range(total_stages):
            if age < egg_stage_duration:
                bar_colors.append('#f7d060')  # Egg color
            elif age < egg_stage_duration + nymphal_stage_duration:
                bar_colors.append('#ff6e40')  # Nymph color
            else:
                bar_colors.append('#5d5d5d')  # Adult color
        
        adult_stage_start = egg_stage_duration + nymphal_stage_duration
        
        # Highlight reproduction days
        for age in range(total_stages):
            if age in [12, 17, 22, 27]:  # Reproduction days
                bar_colors[age] = '#e74c3c'  # Highlight reproduction days
        
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
        ax4.axhline(y=egg_stage_duration + nymphal_stage_duration - 0.5, color='k', linestyle='--', alpha=0.3)
        
        st.pyplot(fig4)
    
        st.download_button(
            label="Download Age Structure Plot",
            data=fig_to_bytes(fig4),
            file_name=f"cockroach_age_structure_day_{selected_day}.png",
            mime="image/png"
        )
        
        # UNIQUE INTERPRETATION FOR AGE STRUCTURE - BASED ON BLACK & MOORE CHAPTER
        st.markdown("""
        **Cockroach Population Age Structure:**
        
        This visualization displays the detailed age distribution of the cockroach population, similar to the Leslie-Lewis
        matrix approach shown in Figure 24.8 of Black & Moore. The horizontal bars represent specific 
        day-age classes, providing insights into cockroach population dynamics:
        
        1. **Reproductive Timing**: The red bars indicate ages when females produce oothecae (egg cases). Similar to 
           the pattern shown in Figure 24.7A, cockroaches show distinct reproductive pulses, creating cohorts 
           that move through the population together.
        
        2. **Development Bottlenecks**: Transitions between life stages (marked by dotted lines) can represent 
           vulnerable periods in the cockroach life cycle. As noted in the text, control strategies can be
           optimized by targeting specific life stages based on their survival rates.
        
        3. **Cohort Identification**: The distinct "waves" of individuals at specific ages indicates separate 
           cohorts moving through the population. As discussed by Black & Moore, understanding these age-structured
           dynamics is crucial for effective pest management.
        
        The model uses daily age classes rather than simple life stages to accurately track individuals through time,
        which as the authors note is essential for precise population models.
        """)
        
        # Create a cohort survival curve
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
                elif age < egg_stage_duration + nymphal_stage_duration:  # Now a nymph
                    individuals = results[day, age]
                    cohort_data.append(("Nymph", age+1-egg_stage_duration, individuals))
                elif age < total_stages:  # Now an adult
                    individuals = results[day, age]
                    cohort_data.append(("Adult", age+1-(egg_stage_duration+nymphal_stage_duration), individuals))
                    
            # Create DataFrame for the cohort
            cohort_df = pd.DataFrame(cohort_data, columns=["Stage", "Age", "Count"])
            
            # Calculate survival rate relative to initial eggs
            cohort_df["Survival Rate"] = cohort_df["Count"] / initial_eggs * 100
            
            # Plot cohort survival
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            
            # Get unique stages in the cohort data
            stages = cohort_df["Stage"].unique()
            stage_colors = {'Egg': '#f7d060', 'Nymph': '#ff6e40', 'Adult': '#5d5d5d'}
            
            for stage in stages:
                stage_data = cohort_df[cohort_df["Stage"] == stage]
                ax5.plot(stage_data.index + cohort_day, stage_data["Survival Rate"], 
                         label=stage, color=stage_colors[stage], linewidth=2, marker='o')
            
            ax5.set_xlabel('Day', fontsize=12)
            ax5.set_ylabel('Survival Rate (%)', fontsize=12)
            ax5.set_
