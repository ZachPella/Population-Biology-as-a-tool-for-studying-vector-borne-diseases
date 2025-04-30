import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io

def run():    
    # Display title and description with academic context
    st.title("🦟 Leslie Matrix Mosquito Population Model")
    st.markdown("#### 🌡️ A discrete, age-structured model of population growth based on principles from Black & Moore")
    
    # Add information about the academic context based on provided readings
    st.markdown("""
    This interactive application simulates mosquito population dynamics using a Leslie Matrix model as described 
    in population biology studies of vector-borne diseases (Black & Moore). The model demonstrates how age structure, 
    survivorship, and fecundity affect vectorial capacity - a critical component in disease transmission.
    
    ### Model Concepts from Literature:
    - **Leslie Matrix Model**: A discrete, age-structured approach representing different life stages
    - **Vectorial Capacity**: The average number of potentially infective bites delivered by vectors feeding on a host in one day
    - **Stable Age Distribution (SAD)**: The constant proportion of individuals in each age class that emerges over time
    """)
    
    # Create sidebar with parameters
    st.sidebar.header("Model Parameters")
    
    # Survival rates with literature context
    st.sidebar.subheader("Survival Rates (p)")
    st.sidebar.markdown("""
    *In Macdonald's equation for vectorial capacity, daily survivorship (p) 
    is one of the most influential parameters, with small changes causing dramatic effects on 
    vector population dynamics.*
    """)
    
    egg_survival = st.sidebar.slider("Egg daily survival rate", 0.0, 1.0, 0.5, 0.01, 
                                    help="As described in Table 24.2, survival rates for eggs typically range from 0.4-0.6")
    larval_survival = st.sidebar.slider("Larval daily survival rate", 0.0, 1.0, 0.6, 0.01,
                                       help="Larvae typically have higher survival than eggs but lower than adults in mosquito species")
    adult_survival = st.sidebar.slider("Adult daily survival rate", 0.0, 1.0, 0.9, 0.01, 
                                      help="Adult survival is typically 0.8-0.95 for mosquito species as described in the literature")
    
    # Initial population
    st.sidebar.subheader("Population Parameters")
    initial_population = st.sidebar.number_input("Initial population (adults at day 8)", 1, 10000, 100, 
                                               help="Starting population of adult mosquitoes")
    
    # Fecundity values based on gonotrophic cycles
    st.sidebar.subheader("Fecundity Values (f)")
    st.sidebar.markdown("""
    *Fecundity is modeled after gonotrophic cycles described in Fig 24.7A. 
    Mosquitoes typically have multiple gonotrophic cycles with declining fecundity over time.*
    """)
        
    fecundity_1 = st.sidebar.number_input("Fecundity after first blood meal (day 12)", 0, 500, 120, 
                                         help="Number of eggs produced after first blood meal")
    fecundity_2 = st.sidebar.number_input("Fecundity after second blood meal (day 17)", 0, 500, 100, 
                                         help="Number of eggs produced after second blood meal")
    fecundity_3 = st.sidebar.number_input("Fecundity after third blood meal (day 22)", 0, 500, 80, 
                                         help="Number of eggs produced after third blood meal")
    fecundity_4 = st.sidebar.number_input("Fecundity after fourth blood meal (day 27)", 0, 500, 60, 
                                         help="Number of eggs produced after fourth blood meal")
    
    # Time periods to simulate
    num_days = st.sidebar.slider("Number of days to simulate", 28, 365, 100, 
                               help="Length of simulation in days")
    
    # Developmental stages for mosquitoes with literature context
    st.sidebar.subheader("Life Stage Durations")
    st.sidebar.markdown("""
    *As noted in Table 24.2, mosquitoes have specific developmental periods for each life stage,
    with age classes critical to understanding population dynamics.*
    """)
        
    egg_stage_duration = st.sidebar.slider("Egg stage duration (days)", 1, 15, 2, 
                                         help="Duration of egg development before hatching")
    larval_stage_duration = st.sidebar.slider("Larval stage duration (days)", 1, 30, 10, 
                                            help="Duration of larval and pupal stages before emerging as adults")
    
    # Options for population growth model
    st.sidebar.subheader("Model Features")
    enable_density_dependence = st.sidebar.checkbox("Enable Density Dependence", False, 
                                                  help="Implement logistic growth as described in Figure 24.5")
    if enable_density_dependence:
        carrying_capacity = st.sidebar.number_input("Carrying Capacity (K)", 100, 1000000, 50000, 
                                                 help="Maximum sustainable population size")
    
    enable_immigration = st.sidebar.checkbox("Enable Immigration & Mortality", False, 
                                           help="Allow population immigration/emigration as in Figure 24.3")
    if enable_immigration:
        immigration_rate = st.sidebar.slider("Immigration rate (new adults per 2 days)", 0, 100, 10, 
                                          help="Number of new individuals entering population")
        mortality_rate = st.sidebar.slider("Mortality rate (fraction of cases leaving)", 0.0, 1.0, 0.1, 
                                        help="Fraction of cases leaving population through mortality")
    
    # Create a helper function to detect development stages
    def get_stage(day):
        if day < egg_stage_duration:
            return "Egg"
        elif day < egg_stage_duration + larval_stage_duration:
            return "Larva"
        else:
            return "Adult"
    
    def run_leslie_model(egg_survival, larval_survival, adult_survival, 
                        initial_pop, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
                        num_days, egg_stage_duration, larval_stage_duration,
                        enable_density_dependence=False, carrying_capacity=50000,
                        enable_immigration=False, immigration_rate=10, mortality_rate=0.1):
        """
        Run the Leslie Matrix population model simulation for mosquitoes based on Lewis-Leslie model
        described in Black & Moore's chapter.
        
        Parameters:
        - egg_survival: Daily survival rate for eggs
        - larval_survival: Daily survival rate for larvae
        - adult_survival: Daily survival rate for adults
        - initial_pop: Initial adult population
        - fecundity_1-4: Number of eggs laid after each blood meal
        - num_days: Number of days to simulate
        - egg_stage_duration: Number of days in egg stage
        - larval_stage_duration: Number of days in larval stage
        - enable_density_dependence: Whether to implement logistic growth
        - carrying_capacity: Maximum sustainable population size (K)
        - enable_immigration: Allow immigration and mortality
        - immigration_rate: Number of adults entering population 
        - mortality_rate: Fraction of cases leaving population
        
        Returns:
        - Population matrix and summary data
        """
        adult_stage_start = egg_stage_duration + larval_stage_duration
        total_stages = max(28, adult_stage_start + 20)  # Ensure we have enough stages for development
        
        # Create the Leslie Matrix (M matrix in Fig 24.8)
        leslie_matrix = np.zeros((total_stages, total_stages))
        
        # Set survival probabilities (subdiagonal) - comparable to the P values in Table 24.2
        for i in range(total_stages-1):
            if i < egg_stage_duration:  # Egg stage
                leslie_matrix[i+1, i] = egg_survival
            elif i < egg_stage_duration + larval_stage_duration:  # Larval stage
                leslie_matrix[i+1, i] = larval_survival
            else:  # Adult stage
                leslie_matrix[i+1, i] = adult_survival
        
        # Set fecundity values (first row) - comparable to the F values in Table 24.2
        # Reproduction occurs on specific adult days (after blood meals)
        reproduction_days = [
            12,  # First blood meal
            17,  # Second blood meal
            22,  # Third blood meal
            27   # Fourth blood meal
        ]
        
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
                
                # Calculate how many adults die (similar to mortality in Fig 24.3)
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
        enable_density_dependence, carrying_capacity if enable_density_dependence else 50000,
        enable_immigration, immigration_rate if enable_immigration else 10, 
        mortality_rate if enable_immigration else 0.1
    )
    
    # Process results for visualization - similar to data processing in Fig 24.9 and 24.10
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
    
    # Display the overall statistics
    st.header("Population Summary")
    
    # Add vectorial capacity interpretation at the top
    vectorial_capacity = adults[-1] * (adult_survival ** egg_stage_duration) * 0.1  # Simplified VC calculation
    r0_estimate = np.log(total_population[-1] / initial_population) / num_days if total_population[-1] > 0 else 0
    
    st.info(f"""
    **Vectorial Capacity Estimate:** {vectorial_capacity:.2f}
    
    Based on Black & Moore's discussion, vectorial capacity is the product of vector density, 
    survival through the extrinsic incubation period, and biting rate. This simplified estimate 
    shows the potential disease transmission capacity of this population.
    
    **Growth Rate (r):** {r0_estimate:.4f}
    
    Comparable to the intrinsic rate of increase (r) in exponential growth models described in the readings.
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Number - Eggs", f"{eggs[-1]:.0f}")
        
    with col2:
        st.metric("Final Number - Larvae", f"{larvae[-1]:.0f}")
        
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Population Trends", 
        "Stage Distribution", 
        "Age Structure", 
        "Data Table",
    ])
    
    with tab1:
        st.header("Population Growth Over Time")
        
        # Academic interpretation for this section
        st.markdown("""
        **Interpretation**: This visualization shows population growth patterns similar to those in 
        Figures 24.1-24.3 of the readings. The relative growth of different life stages reflects survival 
        rates and developmental times. Note how density dependence and immigration/emigration can create 
        cyclical patterns comparable to those seen in field observations of vector populations.
        """)
        
        # Population trend plot, similar to the figures 24.1, 24.2, and 24.3
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
        
        # Population growth rate plot, representing r from the exponential growth model
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        growth_rates = np.zeros(num_days)
        for i in range(1, num_days):
            growth_rates[i] = (total_population[i] / total_population[i-1] - 1) * 100 if total_population[i-1] > 0 else 0
        
        ax2.plot(days[1:], growth_rates[1:], color='#2ca02c', linewidth=2)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Day', fontsize=12)
        ax2.set_ylabel('Growth Rate (%)', fontsize=12)
        ax2.set_title('Daily Population Growth Rate (comparable to r in the readings)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add interpretation of growth rate
        mean_growth = np.mean(growth_rates[1:])
        st.markdown(f"""
        **Mean Growth Rate: {mean_growth:.2f}%**
        
        This growth rate represents the discrete-time analog of the intrinsic rate of increase (r) 
        discussed in the exponential growth model (equation 14) from the readings. When positive, the population 
        grows; when negative, it declines.
        """)
        
        st.pyplot(fig2)
        
        st.download_button(
            label="Download Growth Rate Plot",
            data=fig_to_bytes(fig2),
            file_name="mosquito_growth_rate.png",
            mime="image/png"
        )
    
    with tab2:
        st.header("Stage Distribution Analysis")
        
        # Add academic interpretation
        st.markdown("""
        **Stable Age Distribution Analysis:**
        
        This chart demonstrates the concept of Stable Age Distribution (SAD) described in the readings. 
        Note how the proportions of each life stage eventually stabilize, indicating the population 
        has reached its characteristic age structure. In the Lewis-Leslie model, 
        this equilibrium emerges naturally from the survival and fecundity parameters.
        """)
        
        # Create a stacked area chart for stage proportions - comparable to the stable age distribution in Fig 24.11
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
        
        # Create a pie chart for the final day - representing final stable age distribution
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
                
            ax3.set_title(f'Population Composition on Day {num_days} (SAD)', fontsize=14)
            st.pyplot(fig3)
            
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
        
        # Academic interpretation
        st.markdown("""
        **Age Structure Interpretation:**
        
        This visualization displays the discrete age classes that form the basis of the Leslie Matrix model,
        similar to the n vector shown in the readings. The horizontal bars represent the number of individuals
        in each age class, with colors indicating life stages. Note the reproductive ages (highlighted in red)
        which correspond to the days when adults take blood meals and produce eggs.
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
        
        # Create age structure plot for the selected day - similar to Fig 24.8 n vector
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        
        # Get age distribution for the selected day
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
        
        # Color bars by stage - this creates a visual similar to the M matrix in Fig 24.8
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
        
        st.download_button(
            label="Download Age Structure Plot",
            data=fig_to_bytes(fig4),
            file_name=f"mosquito_age_structure_day_{selected_day}.png",
            mime="image/png"
        )
        
        # Create a cohort survival curve - aligns with survivorship curves in Fig 24.6
        st.subheader("Cohort Survival Analysis")
        
        # Academic interpretation
        st.markdown("""
        **Survivorship Analysis:**
        
        This survival curve tracks a cohort of individuals from egg to adult stage, showing
        mortality patterns similar to those described in the survivorship curve classifications (Type I, II, or III).
        Mosquitoes typically follow a Type II curve with relatively constant mortality rates, though
        the exact pattern depends on the survival rates chosen for each life stage.
        """)
        
        # Use simpler slider for cohort day selection
        cohort_day = st.slider("Select day to start tracking a cohort:", 1, max(1, num_days-10), 1, 
                             help="Similar to survivorship curves in the readings")
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
            ax5.set_title(f'Cohort Survival from Day {cohort_day} (Comparable to the readings)', fontsize=14)
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
            
            # Determine survivorship curve type based on patterns in readings
            survival_pattern = "Type II"  # Default
            
            # Check survival rate pattern to determine type (I, II, or III)
            early_survival = cohort_df["Survival Rate"].iloc[:min(5, len(cohort_df))]
            late_survival = cohort_df["Survival Rate"].iloc[-min(5, len(cohort_df)):]
            
            if early_survival.mean() > 80 and late_survival.mean() < 20:
                survival_pattern = "Type I"  # High early survival, sharp decline in old age (tsetse flies)
            elif np.all(np.diff(cohort_df["Survival Rate"]) < 0) and np.std(np.diff(cohort_df["Survival Rate"])) < 5:
                survival_pattern = "Type II"  # Constant mortality rate (many mosquito species)
            elif early_survival.mean() < 30 and cohort_df["Survival Rate"].iloc[min(10, len(cohort_df)-1)] < 10:
                survival_pattern = "Type III"  # High early mortality (ticks, many egg-laying species)
            
            st.markdown(f"""
            **Survivorship Curve Analysis:**
            
            This cohort follows a **{survival_pattern}** survivorship pattern as described in the readings.
            
            - **Type I**: Low early mortality, death concentrated at old age (like tsetse flies)
            - **Type II**: Constant mortality rate through life (common in many mosquito species)
            - **Type III**: High early mortality, few reaching maturity (ticks, producing many eggs)
            
            The daily survival rates you've chosen ({egg_survival} for eggs, {larval_survival} for larvae, {adult_survival} for adults)
            directly shape this survivorship curve. These survival patterns are major determinants of vectorial capacity and transmission dynamics.
            """)
            
            # Calculate final survival rate
            final_survival = cohort_df["Survival Rate"].iloc[-1] if len(cohort_df) > 0 else 0
            
            # Calculate estimated daily survival rate
            if len(cohort_df) > 1:
                # Use logarithmic regression to estimate daily survival (p) as in equations from the reading
                days = np.array(range(len(cohort_df)))
                survival = np.array(cohort_df["Survival Rate"])
                survival_positive = np.maximum(survival, 0.001)  # Avoid log(0)
                
                # Fit log model: log(p) = log(m)/d from equations in the readings
                log_survival = np.log(survival_positive/100)
                try:
                    # Simple linear regression on log values to get daily survival
                    slope, _ = np.polyfit(days, log_survival, 1)
                    estimated_daily_survival = np.exp(slope)
                    
                    st.markdown(f"""
                    **Estimated Daily Survival Rate:** {estimated_daily_survival:.4f}
                    
                    This estimate is derived from the cohort survival curve and represents the average daily
                    probability of survival across all stages. This is comparable to the
                    daily survivorship (p) value that appears in the vectorial capacity equations from the readings.
                    """)
                except:
                    pass
        else:
            st.warning(f"No eggs were laid on day {cohort_day}. Please select a different day.")
    
    with tab4:
        st.header("Detailed Data")
        
        # Leslie matrix visualization based on readings
        st.subheader("Leslie Matrix Structure (M Matrix)")
        
        st.markdown("""
        This visualization represents the Leslie Matrix (M) as shown in the readings. 
        The matrix combines:
        
        - **Survival probabilities** on the subdiagonal (movement from one age class to the next)
        - **Fecundity values** in the first row (reproduction from reproductive adults to new eggs)
        
        The matrix multiplication M × n₍ₜ₎ = n₍ₜ₊₁₎ projects the age-structured population forward in time.
        """)
        
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
        fig6, ax6 = plt.subplots(figsize=(10, 8))
        im = ax6.imshow(leslie_matrix, cmap='viridis')
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
        
        # Show the detailed data table - life table data similar to the readings
        st.subheader("Population Data by Day")
        
        st.markdown("""
        The data tables below represent the population dynamics over time, formatted in a way 
        that resembles the life tables described in the readings. These tables combine:
        
        - **Age-specific survival rates** (daily survivorship)
        - **Stage-specific population numbers** (eggs, larvae, adults)
        - **Stage distribution** (proportion of population in each stage)
        
        These data can be used to calculate vectorial capacity components and estimate 
        disease transmission potential of the modeled mosquito population.
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
                file_name="mosquito_leslie_matrix_summary.csv",
                mime="text/csv"
            )
        else:
            # Create a detailed dataframe with day-by-age data
            detailed_columns = []
            for i in range(total_stages):
                if i < egg_stage_duration:
                    detailed_columns.append(f'Egg {i+1}')
                elif i < egg_stage_duration + larval_stage_duration:
                    detailed_columns.append(f'Larva {i-egg_stage_duration+1}')
                else:
                    detailed_columns.append(f'Adult {i-(egg_stage_duration+larval_stage_duration)+1}')
            
            detailed_df = pd.DataFrame(results, columns=detailed_columns)
            detailed_df.insert(0, 'Day', days)
            detailed_df['Eggs'] = eggs
            detailed_df['Larvae'] = larvae
            detailed_df['Adults'] = adults
            detailed_df['Total'] = total_population
            
            st.dataframe(detailed_df.style.background_gradient(cmap='viridis', subset=['Total']))
            
            # Download button for detailed CSV
            detailed_csv = detailed_df.to_csv(index=False)
            st.download_button(
                label="Download Detailed Data as CSV",
                data=detailed_csv,
                file_name="mosquito_leslie_matrix_detailed.csv",
                mime="text/csv"
            )
    
    # Add section on vector control implications
    st.header("Vector Control Implications")
    
    st.markdown("""
    ### Application to Vector-Borne Disease Control
    
    The Leslie Matrix model provides important insights for vector control strategies:
    
    1. **Life Stage Targeting**: Different control methods target specific life stages (larvicides vs. adulticides)
    2. **Efficacy Estimation**: Models help predict population responses to control measures
    3. **Optimal Timing**: Identifying when interventions will have maximum impact
    4. **Economic Thresholds**: Determining minimum control efforts needed to reduce disease transmission
    """)
    
    # Create comparative plot showing effects of different control strategies
    # Create data for three scenarios - baseline, larval control, adult control
    control_days = np.arange(1, 101)
    
    # Scenario 1: Baseline (current parameters)
    baseline_adults = adults[:min(100, len(adults))]
    if len(baseline_adults) < 100:
        baseline_adults = np.pad(baseline_adults, (0, 100-len(baseline_adults)), 'constant', constant_values=baseline_adults[-1] if len(baseline_adults) > 0 else 0)
    
    # Scenario 2: Simulated larval control (50% reduction in larval survival)
    larval_control_results, _, _, _ = run_leslie_model(
        egg_survival, larval_survival * 0.5, adult_survival, 
        initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
        min(100, num_days), egg_stage_duration, larval_stage_duration,
        enable_density_dependence, carrying_capacity if enable_density_dependence else 50000,
        enable_immigration, immigration_rate if enable_immigration else 10, 
        mortality_rate if enable_immigration else 0.1
    )
    larval_control_adults = np.sum(larval_control_results[:, adult_indices], axis=1)
    if len(larval_control_adults) < 100:
        larval_control_adults = np.pad(larval_control_adults, (0, 100-len(larval_control_adults)), 'constant', constant_values=larval_control_adults[-1] if len(larval_control_adults) > 0 else 0)
    
    # Scenario 3: Simulated adult control (50% reduction in adult survival)
    adult_control_results, _, _, _ = run_leslie_model(
        egg_survival, larval_survival, adult_survival * 0.5, 
        initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
        min(100, num_days), egg_stage_duration, larval_stage_duration,
        enable_density_dependence, carrying_capacity if enable_density_dependence else 50000,
        enable_immigration, immigration_rate if enable_immigration else 10, 
        mortality_rate if enable_immigration else 0.1
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
    ax7.set_title('Effect of Control Strategies on Adult Mosquito Population', fontsize=14)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    
    # Use log scale if numbers get large
    if max(baseline_adults) > 1000 or max(larval_control_adults) > 1000 or max(adult_control_adults) > 1000:
        ax7.set_yscale('log')
        st.info("Using logarithmic scale for y-axis due to large population numbers")
    
    st.pyplot(fig7)
    
    st.markdown("""
    ### Interpretation of Control Strategies
    
    The above graph shows the expected effect of different control strategies:
    
    1. **Baseline**: Current parameters with no intervention
    2. **Larval Control**: 50% reduction in larval survival rate
    3. **Adult Control**: 50% reduction in adult survival rate
    
    These results highlight how targeting different life stages affects the overall population dynamics.
    The comparative effectiveness of these strategies depends on the current survival rates of each stage, 
    reproductive patterns, density dependence, and immigration/emigration.
    
    In general, reducing adult survival has a more immediate impact on vectorial capacity because it:
    1. Directly reduces the biting population
    2. Has exponential effects on vector survival through the extrinsic incubation period
    3. Reduces the number of individuals reaching reproductive age
    
    However, in some ecological contexts, larval control may be more sustainable and cost-effective.
    """)
    
    # Add vectorial capacity calculations
    st.header("Estimated Vectorial Capacity")
    
    # Get latest adult population
    final_adults = adults[-1]
    
    # Calculate vectorial capacity using formula from the readings
    # Simplified version of V = [ma²pⁿ]/[-ln p]
    biting_rate = 0.25  # Assume 1 bite every 4 days on average
    vector_competence = 0.5  # Proportion of infectious bites that successfully infect
    extrinsic_incubation = 10  # Days from ingestion to transmission capability
    
    # Calculate components
    m = final_adults / 100  # Mosquitoes per human (assuming 100 humans)
    a = biting_rate
    n = extrinsic_incubation
    p = adult_survival
    
    # Calculate vectorial capacity formula: V = [ma²pⁿ]/[-ln p]
    if p > 0 and p < 1:
        vectorial_capacity = (m * (a**2) * (p**n)) / (-np.log(p))
        
        # Create a table with the components
        vc_data = {
            "Parameter": ["m (mosquito density)", "a (biting rate)", "p (daily survival)", 
                         "n (extrinsic incubation)", "V (vectorial capacity)"],
            "Value": [f"{m:.2f}", f"{a:.2f}", f"{p:.2f}", f"{n}", f"{vectorial_capacity:.4f}"],
            "Description": [
                "Mosquitoes per human",
                "Human bites per mosquito per day",
                "Probability of mosquito surviving one day",
                "Days from ingestion to transmission capability",
                "Potential infective bites from a case"
            ]
        }
        
        vc_df = pd.DataFrame(vc_data)
        st.table(vc_df)
        
        st.markdown(f"""
        ### Interpretation of Vectorial Capacity
        
        The calculated vectorial capacity value of **{vectorial_capacity:.4f}** represents the average number of 
        potentially infective bites that could arise from all mosquitoes that bite an infectious host on a single day.
        
        This value is derived from Macdonald's equation as discussed in the readings, and reflects the combined effects of:
        
        1. **Vector density** (m): More mosquitoes per person increases transmission potential
        2. **Biting frequency** (a): Appears as a squared term because vectors must bite twice (to acquire and transmit)
        3. **Daily survival** (p): Appears both as p^n and in the denominator, making it extremely influential
        4. **Extrinsic incubation** (n): Longer incubation reduces vectorial capacity as fewer mosquitoes survive long enough
        
        The vectorial capacity calculation shows why adult survival rate (p) is the most sensitive parameter in disease 
        transmission - small changes in daily survival produce large changes in vectorial capacity, as seen in 
        the control strategy comparison above.
        """)
    else:
        st.warning("Cannot calculate vectorial capacity with current parameters (p must be between 0 and 1)")

if __name__ == "__main__":
    run()
