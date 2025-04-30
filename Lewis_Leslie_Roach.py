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
        for day in range(1, num_days):
            population = leslie_matrix @ population
            results[day, :] = population
        
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
        
        st.pyplot(fig2)
        
        st.download_button(
            label="Download Growth Rate Plot",
            data=fig_to_bytes(fig2),
            file_name="cockroach_growth_rate.png",
            mime="image/png"
        )
    
    with tab2:
        st.header("Stage Distribution Analysis")
        
        # UNIQUE INTERPRETATION FOR STAGE DISTRIBUTION - COCKROACH FOCUS
        st.markdown("""
        **Cockroach Stage Distribution Dynamics:**
        
        This visualization shows how the cockroach population structure evolves over time, eventually reaching 
        a stable age distribution (SAD) - a fundamental concept in population biology. In cockroach populations, 
        this structural pattern has important implications:
        
        1. **Hidden Infestation Indicators**: The high proportion of eggs and nymphs (often >70% of the total 
           population) explains why cockroach infestations are often much larger than apparent from visible adults. 
           This "population iceberg" effect is critical for pest management planning.
        
        2. **Resilience Mechanism**: The extended development time of nymphs creates a buffer against control 
           measures - even if all adults are eliminated, the nymphal reservoir ensures population recovery. 
           This resilience mechanism is similar to the "demographic momentum" described in population models.
        
        3. **Control Strategy Implications**: The eventual stable proportion of each life stage determines what 
           fraction of the population can be targeted by stage-specific control methods (e.g., growth regulators 
           vs. adult baits). Effective management requires addressing all life stages present in the stable 
           distribution.
        """)
        
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
            
            # UNIQUE INTERPRETATION FOR FINAL PROPORTIONS - COCKROACH FOCUS
            st.markdown(f"""
            **Final Population Structure Analysis:**
            
            The pie chart shows the stable age distribution reached after {num_days} days. This equilibrium 
            structure is determined by:
            
            1. **Stage-specific survival rates**: Eggs ({egg_survival:.2f}), Nymphs ({nymphal_survival:.2f}), Adults ({adult_survival:.2f})
            2. **Stage durations**: Egg development ({egg_stage_duration} days), Nymphal development ({nymphal_stage_duration} days)
            3. **Reproductive pattern**: Oviposition occurring at days 12, 17, 22, and 27 of adult life
            
            Cockroach populations typically show a much higher proportion of nymphs than adults due to their 
            extended developmental period. This creates challenges for control programs, as the majority of the 
            population is in life stages that may be less susceptible to certain control measures and often 
            hide in inaccessible locations.
            
            The stable structure shown here helps explain why cockroach control often requires sustained effort 
            rather than one-time treatments - the age distribution ensures a constant supply of developing 
            individuals to replace eliminated adults.
            """)
            
            st.download_button(
                label="Download Population Composition Plot",
                data=fig_to_bytes(fig3),
                file_name="cockroach_population_composition.png",
                mime="image/png"
            )
        else:
            st.warning("No individuals found in the final day to create pie chart.")
    
    with tab3:
        st.header("Age Structure Analysis")
        
        # UNIQUE INTERPRETATION FOR AGE STRUCTURE - COCKROACH FOCUS
        st.markdown("""
        **Cockroach Population Age Structure:**
        
        This visualization displays the detailed age distribution of the cockroach population, revealing patterns 
        that are not apparent in the aggregated life stage counts. The horizontal bars represent specific 
        day-age classes, providing insights into cockroach population dynamics:
        
        1. **Reproductive Timing**: The red bars indicate ages when females produce oothecae (egg cases). Unlike 
           species with continuous reproduction, cockroaches show distinct reproductive pulses, creating cohorts 
           that move through the population together.
        
        2. **Development Bottlenecks**: Transitions between life stages (marked by dotted lines) can represent 
           vulnerable periods in the cockroach life cycle. For example, nymphs undergoing molting may be more 
           susceptible to certain control measures, creating windows of opportunity for management interventions.
        
        3. **Cohort Identification**: The distinct "waves" of individuals at specific ages indicates separate 
           cohorts moving through the population. In pest management, identifying these cohort patterns can help 
           time control measures for maximum effectiveness.
        
        The discrete age structure shown here is particularly important for understanding German cockroach dynamics, 
        as this species carries its oothecae until just before hatching rather than depositing them earlier.
        """)
        
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
        
        # Create a cohort survival curve
        st.subheader("Cohort Survival Analysis")
        
        # UNIQUE INTERPRETATION FOR COHORT SURVIVAL - COCKROACH FOCUS
        st.markdown("""
        **Cockroach Cohort Survival Patterns:**
        
        This survival curve tracks eggs from deposition through development, revealing mortality patterns 
        characteristic of cockroach populations. The shape of this curve provides insights into cockroach 
        life history strategies:
        
        1. **Survival Strategy**: Cockroaches typically exhibit a Type II survivorship curve with relatively 
           constant mortality rates across life stages. This differs from many insect species that show high 
           early mortality (Type III). The protected development of eggs within oothecae contributes to this pattern.
        
        2. **Stage-Specific Vulnerability**: The vertical lines mark transitions between life stages, often 
           associated with changes in survival probability. For cockroaches, the transition from protected eggs 
           to mobile nymphs can be a period of increased risk.
        
        3. **Management Implications**: Control strategies that alter this survivorship curve by increasing 
           mortality at specific life stages can effectively suppress population growth. For example, bait 
           formulations specifically targeting nymphs could steepen the curve during that life stage.
        
        4. **Resistance Development**: The shape of this curve also influences how quickly resistance to control 
           measures can evolve - populations with higher adult survival rates have more opportunities for 
           selection of resistant individuals.
        """)
        
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
            ax5.set_title(f'Cohort Survival from Day {cohort_day}', fontsize=14)
            ax5.legend(fontsize=10)
            ax5.grid(True, alpha=0.3)
            
            # Use log scale for y-axis to match survivorship curves
            ax5.set_yscale('log')
            
            # Add stage transition lines
            transition_days = [
                cohort_day + egg_stage_duration - 1, 
                cohort_day + egg_stage_duration + nymphal_stage_duration - 1
            ]
            for day in transition_days:
                if day < cohort_day + len(cohort_df):
                    ax5.axvline(x=day, color='k', linestyle='--', alpha=0.5)
            
            st.pyplot(fig5)
            
            st.download_button(
                label="Download Cohort Survival Plot",
                data=fig_to_bytes(fig5),
                file_name="cockroach_cohort_survival.png",
                mime="image/png"
            )
            
            # Determine survivorship curve type
            survival_pattern = "Type II"  # Default for most cockroaches
            
            # Check survival rate pattern
            early_survival = cohort_df["Survival Rate"].iloc[:min(5, len(cohort_df))]
            late_survival = cohort_df["Survival Rate"].iloc[-min(5, len(cohort_df)):]
            
            if early_survival.mean() > 80 and late_survival.mean() < 20:
                survival_pattern = "Type I"
            elif np.all(np.diff(cohort_df["Survival Rate"]) < 0) and np.std(np.diff(cohort_df["Survival Rate"])) < 5:
                survival_pattern = "Type II"
            elif early_survival.mean() < 30 and cohort_df["Survival Rate"].iloc[min(10, len(cohort_df)-1)] < 10:
                survival_pattern = "Type III"
            
            # UNIQUE INTERPRETATION FOR SURVIVORSHIP CLASSIFICATION - COCKROACH FOCUS
            st.markdown(f"""
            **Survivorship Classification:**
            
            Based on the survival curve, this cockroach population follows a **{survival_pattern}** survivorship pattern, 
            characterized by:
            
            - **Early life survival**: {early_survival.mean():.1f}% survival in early stages
            - **Late life survival**: {late_survival.mean():.1f}% survival in later stages
            - **Stage-specific mortality**: Changes at life stage transitions
            
            Cockroaches typically exhibit Type II survivorship curves, with relatively constant mortality 
            rates across all age classes. This pattern differs from many other insects that show high early 
            mortality (Type III) and reflects the protected egg development and resilient life history of 
            cockroaches.
            
            The protective behavior of carrying oothecae until hatching (in German cockroaches) or depositing 
            them in protected locations (American cockroaches) contributes to their higher early-stage survival 
            rates compared to insects with exposed eggs. This survivorship pattern contributes to the notorious 
            resilience of cockroach populations and influences optimal control strategies.
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
                    
                    # UNIQUE INTERPRETATION FOR DAILY SURVIVAL - COCKROACH FOCUS
                    st.markdown(f"""
                    **Daily Survival Rate Analysis:**
                    
                    The estimated average daily survival probability from this cohort analysis is **{estimated_daily_survival:.4f}**.
                    
                    For cockroach populations, daily survival rates have important implications:
                    
                    1. **Control Thresholds**: For sustained population reduction, control measures must reduce 
                       daily survival below a critical threshold where deaths exceed births. Based on this model,
                       a reduction of approximately {((1-estimated_daily_survival/0.95)*100):.1f}% in daily survival 
                       would be needed to halt population growth.
                    
                    2. **Life Stage Targeting**: The stage-specific survival rates you've chosen 
                       (Eggs: {egg_survival:.2f}, Nymphs: {nymphal_survival:.2f}, Adults: {adult_survival:.2f}) 
                       suggest that control efforts focused on {
                           "eggs" if egg_survival > nymphal_survival and egg_survival > adult_survival else 
                           "nymphs" if nymphal_survival > egg_survival and nymphal_survival > adult_survival else
                           "adults"
                       } would have the greatest impact on overall population reduction.
                    
                    3. **Resistance Management**: Higher survival rates create more opportunities for selection 
                       of resistant individuals. Integrated pest management approaches that target different life
                       stages can help prevent resistance development by reducing overall population survival.
                    """)
                except:
                    pass
        else:
            st.warning(f"No eggs were laid on day {cohort_day}. Please select a different day.")
    
    with tab4:
        st.header("Leslie Matrix & Life Table Analysis")
        
        # UNIQUE INTERPRETATION FOR LESLIE MATRIX TAB - COCKROACH FOCUS
        st.markdown("""
        **Cockroach Population Matrix Model:**
        
        The Leslie Matrix is the mathematical engine of this cockroach population model. For cockroach populations, 
        this matrix approach captures several key biological features:
        
        1. **Protected Reproduction**: The first row (fecundity values) shows the pulse reproduction pattern, with 
           reproduction concentrated at specific adult ages corresponding to ootheca production cycles. This contrasts 
           with species that show more continuous reproduction.
        
        2. **Stage-Structured Survival**: The subdiagonal elements represent transitions between age classes with 
           stage-specific survival rates. For cockroaches, these survival rates are often higher than many other 
           insects due to their hardiness and protected development.
        
        3. **Time-Delayed Dynamics**: The matrix structure inherently captures the developmental delays between 
           egg deposition and the emergence of reproductive adults. These delays create the cyclic patterns 
           observed in cockroach population growth.
        
        The Leslie Matrix approach is particularly valuable for cockroach management as it allows prediction of 
        future population structure and identification of critical control points in the life cycle.
        """)
        
        leslie_matrix = np.zeros((total_stages, total_stages))
        
        # Set survival probabilities
        for i in range(total_stages-1):
            if i < egg_stage_duration:
                leslie_matrix[i+1, i] = egg_survival
            elif i < egg_stage_duration + nymphal_stage_duration:
                leslie_matrix[i+1, i] = nymphal_survival
            else:
                leslie_matrix[i+1, i] = adult_survival
        
        # Set fecundity values
        for i, day in enumerate([12, 17, 22, 27]):
            if day < total_stages:
                leslie_matrix[0, day] = [fecundity_1, fecundity_2, fecundity_3, fecundity_4][i]
        
        # Create a heatmap of the Leslie matrix
        fig6, ax6 = plt.subplots(figsize=(10, 8))
        im = ax6.imshow(leslie_matrix, cmap='viridis')
        plt.colorbar(im, ax=ax6, label='Transition Rate')
        
        # Add lines to separate life stages in the matrix
        ax6.axhline(y=egg_stage_duration - 0.5, color='w', linestyle='-')
        ax6.axhline(y=egg_stage_duration + nymphal_stage_duration - 0.5, color='w', linestyle='-')
        ax6.axvline(x=egg_stage_duration - 0.5, color='w', linestyle='-')
        ax6.axvline(x=egg_stage_duration + nymphal_stage_duration - 0.5, color='w', linestyle='-')
        
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
                x_labels.append('Nymph 1')
                y_labels.append('Nymph 1')
            elif i < egg_stage_duration + nymphal_stage_duration:
                x_labels.append(f'Nymph {i-egg_stage_duration+1}')
                y_labels.append(f'Nymph {i-egg_stage_duration+1}')
            elif i == egg_stage_duration + nymphal_stage_duration:
                x_labels.append('Adult 1')
                y_labels.append('Adult 1')
            else:
                x_labels.append(f'Adult {i-(egg_stage_duration+nymphal_stage_duration)+1}')
                y_labels.append(f'Adult {i-(egg_stage_duration+nymphal_stage_duration)+1}')
        
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
            file_name="cockroach_leslie_matrix.png",
            mime="image/png"
        )
        
        # Show the detailed data table
        st.subheader("Population Data by Day")
        
        # UNIQUE INTERPRETATION FOR LIFE TABLE - COCKROACH FOCUS
        st.markdown("""
        **Cockroach Life Table Analysis:**
        
        The data tables below provide a detailed demographic description of the cockroach population over time. 
        For pest management purposes, this life table approach offers several advantages:
        
        1. **Detection Probability**: The life table shows what proportion of the population is visible at any time. 
           Since only adults are typically observed, traditional sampling methods may severely underestimate actual 
           infestation levels by missing eggs and hidden nymphs.
        
        2. **Control Evaluation**: By comparing the age structure before and after control measures, managers can 
           assess whether interventions are affecting all life stages or just the more visible adults. Effective 
           control should reduce numbers across all age classes.
        
        3. **Rebound Prediction**: The age structure data allows prediction of population recovery after control 
           efforts. A high proportion of late-stage nymphs indicates that adult numbers will soon increase even if 
           current adult counts are low.
        
        4. **Seasonal Dynamics**: For species like the American cockroach that show seasonal reproduction peaks, 
           the life table approach helps identify optimal timing for preventive control measures before population 
           explosions occur.
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
                file_name="cockroach_leslie_matrix_summary.csv",
                mime="text/csv"
            )
        else:
            # Create a detailed dataframe with day-by-age data
            detailed_columns = []
            for i in range(total_stages):
                if i < egg_stage_duration:
                    detailed_columns.append(f'Egg {i+1}')
                elif i < egg_stage_duration + nymphal_stage_duration:
                    detailed_columns.append(f'Nymph {i-egg_stage_duration+1}')
                else:
                    detailed_columns.append(f'Adult {i-(egg_stage_duration+nymphal_stage_duration)+1}')
            
            detailed_df = pd.DataFrame(results, columns=detailed_columns)
            detailed_df.insert(0, 'Day', days)
            detailed_df['Eggs'] = eggs
            detailed_df['Nymphs'] = nymphs
            detailed_df['Adults'] = adults
            detailed_df['Total'] = total_population
            
            st.dataframe(detailed_df.style.background_gradient(cmap='viridis', subset=['Total']))
            
            # Download button for detailed CSV
            detailed_csv = detailed_df.to_csv(index=False)
            st.download_button(
                label="Download Detailed Data as CSV",
                data=detailed_csv,
                file_name="cockroach_leslie_matrix_detailed.csv",
                mime="text/csv"
            )
    
    # Add section on management implications
    st.header("Management Implications")
    
    # UNIQUE INTERPRETATION FOR MANAGEMENT SECTION - COCKROACH FOCUS
    st.markdown("""
    ### Cockroach Management Strategy Insights
    
    This population model provides critical insights for developing effective cockroach control strategies. 
    Several key management implications emerge from the Leslie Matrix approach:
    
    1. **Control Timing and Targeting**:
       - **Life Stage Vulnerability**: Different life stages show different susceptibilities to control measures
       - **Early Intervention**: The exponential growth pattern means early detection and control is critical
       - **Sustained Approaches**: The stable age distribution indicates that one-time treatments will be ineffective
    
    2. **Integrated Management Approaches**:
       - **Multiple Control Methods**: Using both growth regulators (targeting nymphs) and adulticides
       - **Habitat Modification**: Reducing carrying capacity through sanitation to limit population growth
       - **Barriers and Exclusion**: Preventing population establishment through physical exclusion
    
    3. **Resistance Management**:
       - **Rotation of Control Agents**: Preventing selection for resistance by using multiple control approaches
       - **Population Reservoirs**: Addressing hidden populations of eggs and nymphs that can harbor resistance genes
       - **Complete Elimination**: Targeting all life stages to prevent selective survival of resistant individuals
    """)
    
    # Create comparative plot for control strategies
    # Create data for three scenarios - baseline, nymph control, adult control
    control_days = np.arange(1, 101)
    
    # Scenario 1: Baseline (current parameters)
    baseline_adults = adults[:min(100, len(adults))]
    if len(baseline_adults) < 100:
        baseline_adults = np.pad(baseline_adults, (0, 100-len(baseline_adults)), 'constant', constant_values=baseline_adults[-1] if len(baseline_adults) > 0 else 0)
    
    # Scenario 2: Simulated nymphal control (50% reduction in nymphal survival)
    nymphal_control_results, _, _, _ = run_leslie_model(
        egg_survival, nymphal_survival * 0.5, adult_survival, 
        initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
        min(100, num_days), egg_stage_duration, nymphal_stage_duration
    )
    nymphal_control_adults = np.sum(nymphal_control_results[:, adult_indices], axis=1)
    if len(nymphal_control_adults) < 100:
        nymphal_control_adults = np.pad(nymphal_control_adults, (0, 100-len(nymphal_control_adults)), 'constant', constant_values=nymphal_control_adults[-1] if len(nymphal_control_adults) > 0 else 0)
    
    # Scenario 3: Simulated adult control (50% reduction in adult survival)
    adult_control_results, _, _, _ = run_leslie_model(
        egg_survival, nymphal_survival, adult_survival * 0.5, 
        initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
        min(100, num_days), egg_stage_duration, nymphal_stage_duration
    )
    adult_control_adults = np.sum(adult_control_results[:, adult_indices], axis=1)
    if len(adult_control_adults) < 100:
        adult_control_adults = np.pad(adult_control_adults, (0, 100-len(adult_control_adults)), 'constant', constant_values=adult_control_adults[-1] if len(adult_control_adults) > 0 else 0)
    
    # Plot the comparison
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    ax7.plot(control_days, baseline_adults, label='Baseline', color='#1e88e5', linewidth=2)
    ax7.plot(control_days, nymphal_control_adults, label='50% Reduction in Nymphal Survival', color='#ff6e40', linewidth=2)
    ax7.plot(control_days, adult_control_adults, label='50% Reduction in Adult Survival', color='#5d5d5d', linewidth=2)
    
    ax7.set_xlabel('Day', fontsize=12)
    ax7.set_ylabel('Number of Adult Cockroaches', fontsize=12)
    ax7.set_title('Effect of Control Strategies on Adult Cockroach Population', fontsize=14)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # Use log scale if numbers get large
    if max(baseline_adults) > 1000 or max(nymphal_control_adults) > 1000 or max(adult_control_adults) > 1000:
        ax7.set_yscale('log')
        st.info("Using logarithmic scale for y-axis due to large population numbers")
    
    st.pyplot(fig7)
    
    # UNIQUE INTERPRETATION FOR CONTROL STRATEGIES - COCKROACH FOCUS
    st.markdown("""
    ### Control Strategy Comparison
    
    The plot above compares how different control strategies affect adult cockroach populations over time. 
    This analysis reveals several important patterns specific to cockroach management:
    
    **1. Adult vs. Nymphal Control Effectiveness:**
    
    - **Adult Control (Gray Line)**: Shows immediate reduction in adult numbers but may have less long-term impact
      due to the reservoir of developing nymphs that will mature into adults
      
    - **Nymphal Control (Orange Line)**: Shows a delayed but potentially more sustained reduction by preventing
      nymphs from reaching reproductive age, effectively breaking the population cycle
    
    **2. Timing Considerations:**
    
    - Adult control measures produce immediate visible results (beneficial for client satisfaction)
    - Nymphal control measures show delayed effectiveness (important for long-term management)
    - Optimal strategy often combines both approaches for both immediate and sustained control
    
    **3. Practical Applications:**
    
    - **Baits with IGRs** (Insect Growth Regulators): Target both adults and prevent nymphal development
    - **Dust formulations**: Effective against harboring areas where eggs and nymphs develop
    - **Integrated approaches**: Combining chemical control with habitat modification and exclusion
    
    The comparative effectiveness of these strategies depends on the specific cockroach species and 
    environmental conditions, but the matrix model provides a framework for predicting outcomes under
    different scenarios.
    """)
    
    # Add conclusions
    st.header("Connecting Population Dynamics to Pest Management")
    
    # UNIQUE CONCLUDING INTERPRETATION - COCKROACH FOCUS
    st.markdown("""
    ### From Population Biology to Practical Pest Management
    
    This Leslie Matrix model bridges theoretical population biology and practical cockroach management by
    demonstrating how age structure, survival rates, and reproductive patterns interact to determine 
    population growth and control effectiveness:
    
    1. **Predictive Power**:
       - The model allows prediction of infestation development over time
       - It reveals the "hidden" component of infestations (eggs and early nymphs)
       - It helps anticipate population rebounds after incomplete control efforts
    
    2. **Management Optimization**:
       - Optimal timing of treatments can be determined based on population structure
       - Resource allocation can be guided by identifying most effective intervention points
       - Treatment frequency can be planned based on predicted population recovery rates
    
    3. **Resistance Prevention**:
       - Understanding population structure helps design strategies that minimize selection for resistance
       - Targeting multiple life stages simultaneously reduces selective pressure on any single stage
       - Monitoring age structure can provide early warning of control failures
    
    4. **Evaluation Metrics**:
       - Success of control programs should be measured not just by adult reduction but by changes in
         population structure across all life stages
       - The stable age distribution concept provides a benchmark for determining when true population
         suppression has been achieved
    
    This model demonstrates how principles from population ecology can directly inform practical pest 
    management, leading to more effective and sustainable cockroach control strategies.
    """)

if __name__ == "__main__":
    run()
