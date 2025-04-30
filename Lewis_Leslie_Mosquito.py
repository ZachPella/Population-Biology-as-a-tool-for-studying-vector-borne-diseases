import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io

# Set page config must be the first Streamlit command
st.set_page_config(layout="wide")

def run():    
    # Display title and description with academic context
    st.title("🦟 Leslie Matrix Mosquito Population Model")
    st.markdown("#### 🧬 A discrete, age-structured population dynamics simulator")
    
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
    
    # Introduction to the Leslie Matrix Model on the main page
    st.markdown("""
    ## Introduction to the Leslie Matrix Model
    
    This interactive application simulates mosquito population dynamics using a Leslie Matrix model as described in Black & Moore's chapter on population biology as a tool for studying vector-borne diseases. The model demonstrates how age structure, survivorship, and fecundity affect vectorial capacity - a critical component in disease transmission.
    """)
    
    # Add styled equation block like in the images
    st.markdown("""
    <div style="background-color: #191D32; padding: 10px; border-radius: 5px;">
    <h3 style="color: white; margin: 0;">Leslie Matrix Equation: n<sub>t+1</sub> = M × n<sub>t</sub></h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Add explanation of the terms like in image 2
    st.markdown("""
    **Where:**
    
    * **n<sub>t</sub>** is the population vector at time t, showing the number of individuals in each age class
    * **M** is the Leslie Matrix containing survival probabilities and fecundity values
    * **n<sub>t+1</sub>** is the resulting population vector for the next time step
    
    **The Model's Logic:**
    
    1. Each row of the Leslie Matrix represents an age class
    2. Survival probabilities appear on the subdiagonal (individuals moving to the next age class)
    3. Fecundity values appear in the first row (reproduction from adult age classes)
    4. As the model iterates, it captures both age structure dynamics and population growth
    5. Eventually, the population reaches a stable age distribution (SAD)
    """, unsafe_allow_html=True)
    
    # Add vectorial capacity equation
    st.markdown("""
    <div style="background-color: #191D32; padding: 10px; border-radius: 5px;">
    <h3 style="color: white; margin: 0;">Vectorial Capacity: V = [ma<sup>2</sup>p<sup>n</sup>]/[-ln(p)]</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Where:**
    * **m** = vector density in relation to hosts
    * **a** = biting rate (humans bitten per mosquito per day)
    * **p** = daily survival rate (especially of adults)
    * **n** = extrinsic incubation period (days from ingestion to transmission)
    
    **Adjustable Parameters:**
    * Survival rates for each life stage (eggs, larvae, adults)
    * Duration of each life stage
    * Fecundity after each blood meal
    * Initial population size
    * Optional density dependence and immigration
    """, unsafe_allow_html=True)
    
    # Run the Leslie matrix simulation model
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
    
    # Summary Statistics in stylized metrics layout
    st.markdown("""
    <div style="background-color: #191D32; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
    <h3 style="color: white; margin: 0;">Summary Statistics</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Peak Adult Population", f"{int(peak_adults):,}")
    with col2:
        st.metric("Time to Peak", f"{peak_time} days")
    with col3:
        st.metric("Final Adult Population", f"{int(adults[-1]):,}")
    with col4:
        st.metric("Vectorial Capacity", f"{vectorial_capacity:.2f}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "Population Trends", 
        "Age Structure", 
        "Vectorial Capacity",
        "Data Table"
    ])
    
    with tab1:
        # Population trends plot and analysis
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
        
        # Stage distribution chart
        st.subheader("Life Stage Distribution Over Time")
        
        # Create stacked area chart for stage proportions
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
        
        # Concise interpretation
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
    
        # CONCISE INTERPRETATION
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
        
    with tab2:
        # Age structure tab
        st.subheader("Age Structure Analysis")
        
        # Let user select which day to focus on using radio buttons
        day_options = [1, max(1, int(num_days/3)), max(1, int(2*num_days/3)), num_days]
        selected_day = st.radio(
            "Select a day to view age structure:",
            day_options,
            format_func=lambda x: f"Day {x}",
            horizontal=True
        )
        
        # Age structure plot
        day_idx = selected_day - 1
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
        ax4.set_title(f'Age Structure on Day {selected_day} (n vector representation)', fontsize=14)
        
        # Add stage dividers
        ax4.axhline(y=egg_stage_duration - 0.5, color='k', linestyle='--', alpha=0.3)
        ax4.axhline(y=egg_stage_duration + larval_stage_duration - 0.5, color='k', linestyle='--', alpha=0.3)
        
        st.pyplot(fig4)
        
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
            
            # Calculate survival rate relative to initial eggs - similar to survivorship curves
            cohort_df["Survival Rate"] = cohort_df["Count"] / initial_eggs * 100
            
            # Plot cohort survival with log scale (similar to Fig 24.6)
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
            
            # CONCISE INTERPRETATION
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
            
        else:
            st.warning(f"No eggs were laid on day {cohort_day}. Please select a different day.")
    
    with tab3:
        st.header("Vectorial Capacity Analysis")
        
        # DISPLAY FORMULA AND CALCULATION
        st.markdown("""
        ### Macdonald's Equation for Vectorial Capacity
        
        The vectorial capacity (V) represents the number of potentially infectious bites that would arise from mosquitoes 
        biting a single infectious host for one day. From Black & Moore's chapter (Equation 13):
        
        $V = \\frac{ma^2p^n}{-ln(p)}$
        
        Where:
        - m = vector density in relation to hosts
        - a = biting rate (humans bitten per mosquito per day)
        - p = daily survival rate
        - n = extrinsic incubation period (days from ingestion to transmission)
        """)
        
        # Calculate sensitivity to survival rate changes
        survivorship_values = np.linspace(0.5, 0.99, 50)
        vc_values = []
        
        for p in survivorship_values:
            m = adults[-1]/100  # Mosquitoes per human
            a = biting_rate
            n = extrinsic_incubation
            
            if p > 0 and p < 1:
                vc = (m * (a**2) * vector_competence * (p**n)) / (-np.log(p))
                vc_values.append(vc)
            else:
                vc_values.append(0)
        
        # Plot vectorial capacity sensitivity
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        ax6.plot(survivorship_values, vc_values, 'b-', linewidth=2)
        ax6.axvline(x=adult_survival, color='r', linestyle='--', label=f'Current p={adult_survival}')
        
        # Highlight current value
        current_vc = vectorial_capacity
        ax6.plot(adult_survival, current_vc, 'ro', markersize=8)
        
        ax6.set_xlabel('Adult Daily Survival Rate (p)', fontsize=12)
        ax6.set_ylabel('Vectorial Capacity', fontsize=12)
        ax6.set_title('Sensitivity of Vectorial Capacity to Survival Rate', fontsize=14)
        ax6.grid(True)
        ax6.legend()
        
        st.pyplot(fig6)
        
        # Create comparative plot showing effects of different control strategies
        control_days = np.arange(1, 101)
        
        # Scenario 1: Baseline (current parameters)
        baseline_adults = adults[:min(100, len(adults))]
        if len(baseline_adults) < 100:
            baseline_adults = np.pad(baseline_adults, (0, 100-len(baseline_adults)), 'constant', 
                                    constant_values=baseline_adults[-1] if len(baseline_adults) > 0 else 0)
        
        # Scenario 2: Simulated larval control (50% reduction in larval survival)
        larval_control_results, _, _, _ = run_leslie_model(
            egg_survival, larval_survival * 0.5, adult_survival, 
            initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
            min(100, num_days), egg_stage_duration, larval_stage_duration,
            enable_density_dependence, carrying_capacity,
            enable_immigration, immigration_rate, mortality_rate
        )
        larval_control_adults = np.sum(larval_control_results[:, adult_indices], axis=1)
        if len(larval_control_adults) < 100:
            larval_control_adults = np.pad(larval_control_adults, (0, 100-len(larval_control_adults)), 'constant', 
                                          constant_values=larval_control_adults[-1] if len(larval_control_adults) > 0 else 0)
        
        # Scenario 3: Simulated adult control (50% reduction in adult survival)
        adult_control_results, _, _, _ = run_leslie_model(
            egg_survival, larval_survival, adult_survival * 0.5, 
            initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
            min(100, num_days), egg_stage_duration, larval_stage_duration,
            enable_density_dependence, carrying_capacity,
            enable_immigration, immigration_rate, mortality_rate
        )
        adult_control_adults = np.sum(adult_control_results[:, adult_indices], axis=1)
        if len(adult_control_adults) < 100:
            adult_control_adults = np.pad(adult_control_adults, (0, 100-len(adult_control_adults)), 'constant', 
                                         constant_values=adult_control_adults[-1] if len(adult_control_adults) > 0 else 0)
        
        # Plot control strategies
        st.subheader("Vector Control Strategy Comparison")
        fig8, ax8 = plt.subplots(figsize=(10, 6))
        ax8.plot(control_days, baseline_adults, label='Baseline', color='#1e88e5', linewidth=2)
        ax8.plot(control_days, larval_control_adults, label='50% Reduction in Larval Survival', color='#ff6e40', linewidth=2)
        ax8.plot(control_days, adult_control_adults, label='50% Reduction in Adult Survival', color='#5d5d5d', linewidth=2)
        
        ax8.set_xlabel('Day', fontsize=12)
        ax8.set_ylabel('Number of Adult Mosquitoes', fontsize=12)
        ax8.set_title('Effect of Control Strategies on Adult Mosquito Population', fontsize=14)
        ax8.legend(fontsize=10)
        ax8.grid(True, alpha=0.3)
        
        # Use log scale if numbers get large
        if max(baseline_adults) > 1000 or max(larval_control_adults) > 1000 or max(adult_control_adults) > 1000:
            ax8.set_yscale('log')
            st.info("Using logarithmic scale for y-axis due to large population numbers")
        
        st.pyplot(fig8)
        
        # Concise interpretation
        st.markdown("""
        **Vectorial Capacity × Control Strategy Interactions**
        
        The vectorial capacity analysis reveals critical relationships between mosquito population parameters and disease transmission:
        
        * **Exponential p Effect**: A small increase in daily survival (p) from 0.8 to 0.9 causes a ~240% increase in vectorial capacity, demonstrating why adult control is highly effective
        * **Nonlinear Response**: The p^n term in the vectorial capacity equation creates an exponential relationship that accelerates as p approaches 1.0
        * **Comparative Control Efficacy**: Adult control (reducing p) shows immediate impact while larval control (reducing future adults) has delayed effects with similar long-term outcomes
        * **Intervention Thresholds**: The graphs identify critical survival thresholds where vectorial capacity drops below 1.0, preventing disease maintenance
        
        **Control Implications:**
        * Adulticidal measures targeting survival rate provide the most immediate reduction in disease transmission
        * Integrated strategies combining both adult and larval control achieve complementary effects
        * The nonlinear relationship between p and vectorial capacity means even modest reductions in adult survival yield substantial epidemiological benefits
        """)
        
        # Create table with vectorial capacity components
        vc_data = {
            "Parameter": ["m (vector:host ratio)", "a (biting rate)", "b (vector competence)", 
                        "p (daily survival)", "n (extrinsic incubation)", "V (vectorial capacity)"],
            "Value": [f"{adults[-1]/100:.2f}", f"{biting_rate:.2f}", f"{vector_competence:.2f}", 
                    f"{adult_survival:.2f}", f"{extrinsic_incubation}", f"{vectorial_capacity:.4f}"],
            "Description": [
                "Mosquitoes per human",
                "Human bites per mosquito per day",
                "Proportion of vectors that develop infection",
                "Probability of mosquito surviving one day",
                "Days from ingestion to transmission capability",
                "Potential infective bites from a single case"
            ]
        }
        
        vc_df = pd.DataFrame(vc_data)
        st.table(vc_df)
    
    with tab4:
        st.header("Data Tables & Leslie Matrix")
        
        # Create tabs for different data views
        data_tab1, data_tab2 = st.tabs(["Population Summary", "Leslie Matrix Structure"])
        
        with data_tab1:
            # Create a DataFrame for the summary data
            summary_df = pd.DataFrame({
                'Day': days,
                'Eggs': eggs,
                'Larvae': larvae,
                'Adults': adults,
                'Total': total_population,
                '%Eggs': percent_eggs,
                '%Larvae': percent_larvae,
                '%Adults': percent_adults
            })
            
            st.dataframe(summary_df.style.background_gradient(cmap='viridis', subset=['Total']))
            
            # Download button for CSV
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download Summary Data as CSV",
                data=csv,
                file_name="mosquito_leslie_matrix_summary.csv",
                mime="text/csv"
            )
        
        with data_tab2:
            # Display the Leslie Matrix as a heatmap
            leslie_matrix = np.zeros((total_stages, total_stages))
            
            # Set survival probabilities
            for i in range(total_stages-1):
                if i < egg_stage_duration:
                    leslie_matrix[i+1, i] = egg_survival
                elif i < egg_stage_duration + larval_stage_duration:
                    leslie_matrix[i+1, i] = larval_survival
                else:
                    leslie_matrix[i+1, i] = adult_survival
            
            # Set fecundity values
            for i, day in enumerate([12, 17, 22, 27]):
                if day < total_stages:
                    leslie_matrix[0, day] = [fecundity_1, fecundity_2, fecundity_3, fecundity_4][i]
            
            # Create a heatmap of the Leslie matrix
            fig7, ax7 = plt.subplots(figsize=(10, 8))
            im = ax7.imshow(leslie_matrix, cmap='viridis')
            plt.colorbar(im, ax=ax7, label='Transition Rate')
            
            ax7.set_title('Leslie Matrix Visualization', fontsize=14)
            ax7.set_xlabel('Current Age (From)', fontsize=12)
            ax7.set_ylabel('Next Age (To)', fontsize=12)
            
            st.pyplot(fig7)
            
            st.markdown("""
            **Leslie Matrix Structure:**
            
            The Leslie Matrix (M) contains:
            - **First row**: Fecundity values - eggs produced by each age class
            - **Subdiagonal**: Survival probabilities between age classes
            - **Everything else**: Zeros (individuals can only age one day at a time)
            
            As described in Fig 24.8 of Black & Moore, the population vector at time t+1 is calculated by 
            multiplying the Leslie Matrix by the population vector at time t: n(t+1) = M × n(t)
            """)

if __name__ == "__main__":
    run()
