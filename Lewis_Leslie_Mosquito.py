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
    ### Introduction to the Lewis-Leslie Model
    
    This interactive application simulates mosquito population dynamics using a Leslie Matrix model based on 
    population biology principles for studying vector populations.
    Adjust the parameters using the sliders and see how they affect the population growth.
    
    **Definition**: The Leslie Matrix model is a discrete-time, age-structured population model that describes 
    population growth when age-specific survival rates and reproductive rates can be estimated. It's a fundamental 
    mathematical tool in population ecology, representing stage transitions and fecundity in a matrix format.
    
    **Core Concept**: At the heart of the Leslie Matrix model is the projection of current population structure 
    to future time steps through matrix multiplication:
    
    **Leslie Matrix Equation**: 
    $n_{t+1} = M × n_{t}$
    
    Where:
    - $n_{t+1}$ is the population vector at the next time step
    - $M$ is the Leslie Matrix containing survival rates and fecundity values
    - $n_{t}$ is the current population vector by age class
    
    **The Model's Logic**:
    1. Each age class has a specific survival probability to the next age class
    2. Only certain adult age classes reproduce, with age-specific fecundity
    3. The population structure changes over time until reaching a stable age distribution
    4. The matrix eigenvalue determines whether the population grows, shrinks, or stabilizes
    """)
    
    # Create sidebar with parameters
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
                              help="Daily probability an egg survives (typically 0.4-0.6)")
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
                                            help="Implement logistic growth")
    if enable_density_dependence:
        carrying_capacity = st.sidebar.number_input("Carrying Capacity (K)", 100, 1000000, 50000, 
                                           help="Maximum sustainable population size")
    else:
        carrying_capacity = 50000
        
    enable_immigration = st.sidebar.checkbox("Enable Immigration", False, 
                                     help="Allow immigration of mosquitoes")
    if enable_immigration:
        immigration_rate = st.sidebar.slider("Immigration rate (adults per 2 days)", 0, 100, 10)
    else:
        immigration_rate = 0

    # Model code
    def run_leslie_model(egg_survival, larval_survival, adult_survival, 
                        initial_pop, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
                        num_days, egg_stage_duration, larval_stage_duration,
                        enable_density_dependence=False, carrying_capacity=50000,
                        enable_immigration=False, immigration_rate=0):
        """
        Run the Leslie Matrix population model simulation for mosquitoes
        """
        adult_stage_start = egg_stage_duration + larval_stage_duration
        total_stages = max(30, adult_stage_start + 20)  # Ensure we have enough stages
        
        # Create the Leslie Matrix (M matrix)
        leslie_matrix = np.zeros((total_stages, total_stages))
        
        # Set survival probabilities (subdiagonal)
        for i in range(total_stages-1):
            if i < egg_stage_duration:  # Egg stage
                leslie_matrix[i+1, i] = egg_survival
            elif i < egg_stage_duration + larval_stage_duration:  # Larval stage
                leslie_matrix[i+1, i] = larval_survival
            else:  # Adult stage
                leslie_matrix[i+1, i] = adult_survival
        
        # Set fecundity values (first row) - after blood meals on specific days
        reproduction_days = [12, 17, 22, 27]  # Blood meal days from readings
        
        # Ensure indices are valid
        for i, day in enumerate(reproduction_days):
            if day < total_stages:
                leslie_matrix[0, day] = [fecundity_1, fecundity_2, fecundity_3, fecundity_4][i]
        
        # Initialize population vector
        population = np.zeros(total_stages)
        population[adult_stage_start] = initial_pop  # Start with initial adults
        
        # Initialize results matrix
        results = np.zeros((num_days, total_stages))
        results[0, :] = population
        
        # Run the simulation
        for day in range(1, num_days):
            # Apply Leslie matrix to get next day's population
            new_population = leslie_matrix @ population
            
            # Apply density dependence if enabled
            if enable_density_dependence:
                total_population = np.sum(population)
                if total_population > 0:
                    # Apply logistic scaling factor
                    scaling_factor = 1 + ((carrying_capacity - total_population) / carrying_capacity)
                    scaling_factor = max(0.1, min(scaling_factor, 1.5))  # Limit scaling to prevent extremes
                    new_population = new_population * scaling_factor
            
            # Apply immigration if enabled
            if enable_immigration and day % 2 == 0:  # Every 2 days
                # Add new adults from immigration
                new_population[adult_stage_start] += immigration_rate
            
            population = new_population
            results[day, :] = population
        
        return results, total_stages, egg_stage_duration, larval_stage_duration
    
    # Run the model
    results, total_stages, egg_stage_duration, larval_stage_duration = run_leslie_model(
        egg_survival, larval_survival, adult_survival, 
        initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
        num_days, egg_stage_duration, larval_stage_duration,
        enable_density_dependence, carrying_capacity,
        enable_immigration, immigration_rate
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
    
    # Calculate stage percentages for stable age distribution analysis
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
    
    # Summary Statistics
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
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Population Trends", 
        "Stage Distribution", 
        "Age Structure", 
        "Leslie Matrix"
    ])
    
    with tab1:
        st.header("Population Growth Over Time")
        
        # Population trend plot
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
    
        # Population dynamics interpretation
        st.subheader("Interpretation")
    
        st.markdown("""
        **Mosquito Population Growth Dynamics:**
        
        This visualization reveals characteristic growth patterns in mosquito populations, with several important features:
        
        1. **Pulse Reproduction Pattern**: Mosquitoes show distinct pulses of reproduction as females take blood meals at 
           regular intervals, creating the step-like increases in egg numbers. This pattern corresponds directly to the 
           gonotrophic cycles described in the reading.
        
        2. **Stage Duration Effects**: The relative lengths of egg and larval stages create a time-delayed pattern of 
           population growth, with waves of individuals moving through developmental stages. This developmental timing 
           directly influences vectorial capacity.
        
        3. **Control Implications**: For vector management, this growth curve suggests that interventions targeting 
           adults before they reproduce will be most effective at reducing disease transmission.
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
        
        mean_growth = np.mean(growth_rates[1:])
        st.pyplot(fig2)
        
        st.markdown(f"""
        **Growth Rate Pattern Analysis:**
        
        The growth rate plot reveals how mosquito populations expand over time. The mean daily growth rate 
        of {mean_growth:.2f}% translates to significant population increases over short periods.
        
        Key observations:
        
        1. **Cyclical Growth Patterns**: The spikes in growth rate correspond to the maturation of large 
           cohorts of larvae into reproductive adults, followed by blood meals and egg-laying events.
        
        2. **Initial Establishment Phase**: Vector populations typically show a lag phase with lower 
           growth rates followed by a rapid acceleration as multiple reproducing females become established.
        
        3. **Carrying Capacity Effects**: When density dependence is enabled, growth rates would eventually decline 
           as populations approach carrying capacity due to resource limitations.
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
            
            # Stage distribution interpretation
            st.subheader("Interpretation")
            st.markdown("""
            **Mosquito Stage Distribution Dynamics:**
            
            This visualization shows how the mosquito population structure evolves over time, eventually reaching 
            a stable age distribution (SAD) - a fundamental concept in population biology described in the reading. 
            In mosquito populations, this structural pattern has important implications:
            
            1. **Population Structure**: As shown in Figure 24.11 from the reading, mosquito populations typically 
               stabilize with a higher proportion of larvae than adults, creating a characteristic distribution pattern.
            
            2. **Hidden Reservoir Effect**: The high proportion of eggs and larvae explains why mosquito populations 
               can quickly rebound after control efforts targeting only adults. This "population iceberg" effect is 
               critical for vector management planning.
            
            3. **Vectorial Capacity Implications**: The stable age distribution directly affects disease transmission 
               potential by determining what proportion of the population survives to ages capable of transmitting pathogens.
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
        
        # Highlight reproduction days
        for age in range(total_stages):
            if age in [12, 17, 22, 27]:  # Blood meal days from reading
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
        
        # Age structure interpretation
        st.subheader("Interpretation")
        
        st.markdown("""
        **Mosquito Population Age Structure:**
        
        This visualization displays the detailed age distribution of the mosquito population, revealing patterns 
        that are not apparent in the aggregated life stage counts:
        
        1. **Blood Meal Timing**: The red bars indicate ages when females take blood meals and produce eggs. 
           As shown in Table 24.2 from the reading, mosquitoes produce eggs on specific days following blood 
           meals (days 12, 17, 22, and 27), creating distinct reproductive pulses.
        
        2. **Development Bottlenecks**: Transitions between life stages (marked by dotted lines) represent 
           vulnerable periods in the mosquito life cycle, as survival rates differ between stages.
        
        3. **Transmission Implications**: Only adult mosquitoes that survive beyond the extrinsic incubation 
           period can transmit disease, which explains why adult survival rate has such a strong impact on 
           vectorial capacity.
        """)
        
        # Cohort survival analysis
        st.subheader("Cohort Survival Analysis")
        
        # Slider for cohort day selection
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
            
            # Use log scale for y-axis to match survivorship curves
            ax5.set_yscale('log')
            
            # Add stage transition lines
            transition_days = [
                cohort_day + egg_stage_duration - 1, 
                cohort_day + egg_stage_duration + larval_stage_duration - 1
            ]
            for day in transition_days:
                if day < cohort_day + len(cohort_df):
                    ax5.axvline(x=day, color='k', linestyle='--', alpha=0.5)
            
            st.pyplot(fig5)
            
            st.download_button(
                label="Download Cohort Survival Plot",
                data=fig_to_bytes(fig5),
                file_name="mosquito_cohort_survival.png",
                mime="image/png"
            )
            
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
                    
                    # Daily survival analysis
                    st.markdown(f"""
                    **Daily Survival Rate Analysis:**
                    
                    The estimated average daily survival probability from this cohort analysis is **{estimated_daily_survival:.4f}**.
                    
                    According to the reading, daily survival rates are critical components of vectorial capacity.
                    Small changes in survival rates can have large impacts on population growth. The reading explains that
                    survival rates can be measured using mark-release-recapture methods or physiological age grading
                    techniques to track cohorts.
                    
                    For mosquito populations, the stage-specific survival rates you've chosen 
                    (Eggs: {egg_survival:.2f}, Larvae: {larval_survival:.2f}, Adults: {adult_survival:.2f}) 
                    directly influence the population's growth potential.
                    """)
                except:
                    pass
        else:
            st.warning(f"No eggs were laid on day {cohort_day}. Please select a different day.")
    
    with tab4:
        st.header("Leslie Matrix Analysis")
        
        # Leslie Matrix description
        st.markdown("""
        **Mosquito Population Matrix Model:**
        
        The Leslie Matrix is the mathematical engine of this population model. As described in the reading,
        the Leslie-Lewis matrix approach (M × n₍ = n₍₊₁) tracks how a population changes over time by representing:
        
        1. **Age-Structured Survival**: The subdiagonal elements of the matrix represent transition 
           probabilities between age classes with stage-specific survival rates.
        
        2. **Reproductive Timing**: The first row contains fecundity values at specific ages, showing when
           reproduction occurs during the organism's lifespan.
        
        3. **Population Projection**: By multiplying the matrix by the current population vector, we can
           predict the population structure in the next time period.
        """)
        
        # Create Leslie matrix for visualization
        leslie_matrix_vis = np.zeros((total_stages, total_stages))
        
        # Set survival probabilities
        for i in range(total_stages-1):
            if i < egg_stage_duration:
                leslie_matrix_vis[i+1, i] = larval_survival
            else:  # Adult stage
                leslie_matrix_vis[i+1, i] = adult_survival
        
        # Set fecundity values (first row)
        for i, day in enumerate([12, 17, 22, 27]):
            if day < total_stages:
                leslie_matrix_vis[0, day] = [fecundity_1, fecundity_2, fecundity_3, fecundity_4][i]
        
        # Create a heatmap of the Leslie matrix
        fig6, ax6 = plt.subplots(figsize=(10, 8))
        im = ax6.imshow(leslie_matrix_vis, cmap='viridis')
        plt.colorbar(im, ax=ax6, label='Transition Rate')
        
        # Add lines to separate life stages in the matrix
        ax6.axhline(y=egg_stage_duration - 0.5, color='w', linestyle='-')
        ax6.axhline(y=egg_stage_duration + larval_stage_duration - 0.5, color='w', linestyle='-')
        ax6.axvline(x=egg_stage_duration - 0.5, color='w', linestyle='-')
        ax6.axvline(x=egg_stage_duration + larval_stage_duration - 0.5, color='w', linestyle='-')
        
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
                x_labels.append('Larva 1')
                y_labels.append('Larva 1')
            elif i < egg_stage_duration + larval_stage_duration:
                x_labels.append(f'Larva {i-egg_stage_duration+1}')
                y_labels.append(f'Larva {i-egg_stage_duration+1}')
            elif i == egg_stage_duration + larval_stage_duration:
                x_labels.append('Adult 1')
                y_labels.append('Adult 1')
            else:
                x_labels.append(f'Adult {i-(egg_stage_duration+larval_stage_duration)+1}')
                y_labels.append(f'Adult {i-(egg_stage_duration+larval_stage_duration)+1}')
        
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
            file_name="mosquito_leslie_matrix.png",
            mime="image/png"
        )
        
        # Table 24.2 reproduction from the reading
        st.subheader("Life Table Analysis")
        
        st.markdown("""
        **Life Table Example (Based on Table 24.2 from the reading):**
        
        The life table below shows the theoretical mosquito life table parameters that form the basis of this model:
        """)
        
        # Create a dataframe for the life table data based on Table 24.2
        life_table_data = {
    "Life stage": ["Egg", "Larva", "Larva", "Larva", "Larva", "Larva", "Larva", "Pupa", 
                   "Adult", "Adult", "Adult", "Adult", "Adult", "Adult", "Adult", "Adult",
                   "Adult", "Adult", "Adult", "Adult", "Adult", "Adult", "Adult", "Adult",
                   "Adult", "Adult", "Adult", "Adult", "Adult"],
    "Chronological age (days)": list(range(29)),
    "Probability of daily survival (p)": [0.5] + [0.6]*7 + [0.9]*21,  # Changed to 21 adult values to get 29 total
    "Fecundity (f)": [0]*12 + [120] + [0]*4 + [100] + [0]*4 + [80] + [0]*4 + [60] + [0]
}
        
        life_table_df = pd.DataFrame(life_table_data)
        
        # Display the life table
        st.dataframe(life_table_df)
        
        # Add description of Table 24.2
        st.markdown("""
        The life table shows:
        
        - **Egg survivorship** is 0.5 (50% daily survival)
        - **Larval survivorship** is 0.6 (60% daily survival)
        - **Adult survivorship** is 0.9 (90% daily survival)
        - **Fecundity** occurs on days 12, 17, 22, and 27, with decreasing egg output each time
        
        This model follows the mosquito gonotrophic cycle where each female requires a blood meal for
        each batch of eggs, with approximately 5 days between blood meals.
        """)
        
        # Control strategy analysis
        st.header("Vector Control Strategy Analysis")
        
        # Create data for three scenarios
        control_days = np.arange(1, 101)
        
        # Scenario 1: Baseline (current parameters)
        baseline_adults = adults[:min(100, len(adults))]
        if len(baseline_adults) < 100:
            baseline_adults = np.pad(baseline_adults, (0, 100-len(baseline_adults)), 'constant', constant_values=baseline_adults[-1] if len(baseline_adults) > 0 else 0)
        
        # Scenario 2: Larval control (50% reduction in larval survival)
        larval_control_results, _, _, _ = run_leslie_model(
            egg_survival, larval_survival * 0.5, adult_survival, 
            initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
            min(100, num_days), egg_stage_duration, larval_stage_duration
        )
        larval_control_adults = np.sum(larval_control_results[:, adult_indices], axis=1)
        if len(larval_control_adults) < 100:
            larval_control_adults = np.pad(larval_control_adults, (0, 100-len(larval_control_adults)), 'constant', constant_values=larval_control_adults[-1] if len(larval_control_adults) > 0 else 0)
        
        # Scenario 3: Adult control (50% reduction in adult survival)
        adult_control_results, _, _, _ = run_leslie_model(
            egg_survival, larval_survival, adult_survival * 0.5, 
            initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
            min(100, num_days), egg_stage_duration, larval_stage_duration
        )
        adult_control_adults = np.sum(adult_control_results[:, adult_indices], axis=1)
        if len(adult_control_adults) < 100:
            adult_control_adults = np.pad(adult_control_adults, (0, 100-len(adult_control_adults)), 'constant', constant_values=adult_control_adults[-1] if len(adult_control_adults) > 0 else 0)
        
        # Plot the comparison
        fig7, ax7 = plt.subplots(figsize=(10, 6))
        ax7.plot(control_days, baseline_adults, label='Baseline', color='#1e88e5', linewidth=2)
        ax7.plot(control_days, larval_control_adults, label='50% Reduction in Larval Survival', color='#ff6e40', linewidth=2)
        ax7.plot(control_days, adult_control_adults, label='50% Reduction in Adult Survival', color='#5d5d5d', linewidth=2)
        
        ax7.set_xlabel('Day', fontsize=12)
        ax7.set_ylabel('Number of Adult Mosquitoes', fontsize=12)
        ax7.set_title('Effect of Control Strategies on Adult Population', fontsize=14)
        ax7.legend(fontsize=10)
        ax7.grid(True, alpha=0.3)
        
        # Use log scale if numbers get large
        if max(baseline_adults) > 1000 or max(larval_control_adults) > 1000 or max(adult_control_adults) > 1000:
            ax7.set_yscale('log')
            st.info("Using logarithmic scale for y-axis due to large population numbers")
        
        st.pyplot(fig7)
        
        # Control strategies analysis
        st.markdown("""
        ### Control Strategy Comparison
        
        The plot above demonstrates concepts from the reading regarding stage-specific control strategies:
        
        **Adult vs. Larval Control Effectiveness:**
        
        According to Figure 24.13 in the reading, when using interventions to reduce vector populations:
        - Changes in survival rates have different impacts depending on the life stage targeted
        - The reading shows that targeting larvae (immature stages) can be more efficient than targeting adults
          in some circumstances
        - However, the reading notes that in real populations, density-dependent factors may affect these outcomes
        
        **Model Applications for Control:**
        
        As noted in the reading:
        - The Leslie matrix approach allows prediction of how control measures will affect future population structure
        - Vectorial capacity is most strongly affected by adult survival rate, as evident in the exponent in the formula
        - Economic thresholds can be calculated to determine the minimum control effort needed
        """)
        
        # Add concluding section
        st.header("Connecting Population Dynamics to Vector Management")
        
        st.markdown("""
        ### Practical Applications of Population Biology Models
        
        As emphasized in the reading, population biology provides essential tools for understanding and managing vector populations:
        
        1. **Model Limitations**:
           - These models are primarily educational tools
           - Real-world populations are more complex with factors like density dependence
           - Models should be used for comparative analysis rather than absolute predictions
        
        2. **Data Collection Methods**:
           - The reading describes methods for measuring key parameters like survival rates and fecundity
           - Techniques include mark-release-recapture, physiological age grading, and cohort analysis
        
        3. **Comparative Approaches**:
           - The Leslie matrix approach demonstrates how mathematical models can guide practical vector control
           - By identifying which parameters most strongly influence population growth, we can develop targeted interventions
           - The stable age distribution concept helps explain why vector populations can be difficult to control
        """)
        
        # Display acknowledgment
        st.info("This simulation is based on the Leslie-Lewis Matrix model as described in the reading materials. The parameters used match those from Table 24.2 for a theoretical mosquito population.")

if __name__ == "__main__":
    run()
