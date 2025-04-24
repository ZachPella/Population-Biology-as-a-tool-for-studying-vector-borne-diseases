import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io

def run():    
    # Display title and description
    st.title("🦟 Leslie Matrix Mosquitos Population Model")
    st.markdown("####🌡️A discrete, age-structured model of population growth")
    st.markdown("""
    This interactive application simulates mosquito population dynamics using a Leslie Matrix model.
    Adjust the parameters using the sliders and see how they affect the population growth.
    
    **Parameters:**
    - **Egg survival rate**: Daily survival probability for eggs
    - **Larval survival rate**: Daily survival probability for larvae (aquatic stage)
    - **Adult survival rate**: Daily survival probability for adult mosquitoes
    - **Initial population**: Starting number of individuals
    - **Fecundity values**: Number of eggs produced at different adult ages after blood meals
    """)
    
    # Create sidebar with parameters
    st.sidebar.header("Model Parameters")
    
    # Survival rates
    egg_survival = st.sidebar.slider("Egg daily survival rate", 0.0, 1.0, 0.9, 0.01)
    larval_survival = st.sidebar.slider("Larval daily survival rate", 0.0, 1.0, 0.9, 0.01)
    adult_survival = st.sidebar.slider("Adult daily survival rate", 0.0, 1.0, 0.8, 0.01)
    
    # Initial population
    initial_population = st.sidebar.number_input("Initial population (adults at day 8)", 1, 10000, 1000, help="Starting population of adult mosquitoes")
    
    # Fecundity values (based on blood meals in female mosquitoes)
    fecundity_1 = st.sidebar.number_input("Fecundity after first blood meal (day 12)", 0, 500, 120, help="Number of eggs produced after first blood meal")
    fecundity_2 = st.sidebar.number_input("Fecundity after second blood meal (day 17)", 0, 500, 100, help="Number of eggs produced after second blood meal")
    fecundity_3 = st.sidebar.number_input("Fecundity after third blood meal (day 22)", 0, 500, 80, help="Number of eggs produced after third blood meal")
    fecundity_4 = st.sidebar.number_input("Fecundity after fourth blood meal (day 27)", 0, 500, 60, help="Number of eggs produced after fourth blood meal")
    
    # Time periods to simulate
    num_days = st.sidebar.slider("Number of days to simulate", 28, 200, 60, help="Length of simulation in days")
    
    # Developmental stages for mosquitoes
    egg_stage_duration = st.sidebar.slider("Egg stage duration (days)", 1, 15, 2, help="Duration of egg development before hatching")
    larval_stage_duration = st.sidebar.slider("Larval stage duration (days)", 1, 30, 10, help="Duration of larval and pupal stages before emerging as adults")
    
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
                        num_days, egg_stage_duration, larval_stage_duration):
        """
        Run the Leslie Matrix population model simulation for mosquitoes
        
        Parameters:
        - egg_survival: Daily survival rate for eggs
        - larval_survival: Daily survival rate for larvae
        - adult_survival: Daily survival rate for adults
        - initial_pop: Initial adult population
        - fecundity_1-4: Number of eggs laid after each blood meal
        - num_days: Number of days to simulate
        - egg_stage_duration: Number of days in egg stage
        - larval_stage_duration: Number of days in larval stage
        
        Returns:
        - Population matrix and summary data
        """
        adult_stage_start = egg_stage_duration + larval_stage_duration
        total_stages = max(28, adult_stage_start + 20)  # Ensure we have enough stages for development
        
        # Create the Leslie Matrix
        leslie_matrix = np.zeros((total_stages, total_stages))
        
        # Set survival probabilities (subdiagonal)
        for i in range(total_stages-1):
            if i < egg_stage_duration:  # Egg stage
                leslie_matrix[i+1, i] = egg_survival
            elif i < egg_stage_duration + larval_stage_duration:  # Larval stage
                leslie_matrix[i+1, i] = larval_survival
            else:  # Adult stage
                leslie_matrix[i+1, i] = adult_survival
        
        # Set fecundity values (first row)
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
        
        return results, total_stages, egg_stage_duration, larval_stage_duration
    
    # Run the model
    results, total_stages, egg_stage_duration, larval_stage_duration = run_leslie_model(
        egg_survival, larval_survival, adult_survival, 
        initial_population, fecundity_1, fecundity_2, fecundity_3, fecundity_4, 
        num_days, egg_stage_duration, larval_stage_duration
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
    
    # Calculate stage percentages
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
    tab1, tab2, tab3, tab4 = st.tabs(["Population Trends", "Stage Distribution", "Age Structure", "Data Table"])
    
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
            file_name="mosquito_growth_rate.png",
            mime="image/png"
        )
    
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
            title='Relative Proportion of Life Stages Over Time'
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
        ax4.axhline(y=egg_stage_duration + larval_stage_duration - 0.5, color='k', linestyle='--', alpha=0.3)
        
        st.pyplot(fig4)
        
        st.download_button(
            label="Download Age Structure Plot",
            data=fig_to_bytes(fig4),
            file_name=f"mosquito_age_structure_day_{selected_day}.png",
            mime="image/png"
        )
        
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
        else:
            st.warning(f"No eggs were laid on day {cohort_day}. Please select a different day.")
    
    with tab4:
        st.header("Detailed Data")
        
        # Leslie matrix visualization
        st.subheader("Leslie Matrix Structure")
        
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
        
        # Show the detailed data table
        st.subheader("Population Data by Day")
        
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
if __name__ == "__main__":
    run()
