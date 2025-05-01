import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import io
import math

# Try to import plotly
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly is not installed. Some visualizations will not be available. Please add plotly to your requirements.txt file.")


def run():
    # Display title and description
    st.title("🦟 Macdonald's Vectorial Capacity Model")
    st.markdown("#### 🔬 Quantifying vector-borne disease transmission potential")
    st.markdown("""
    This interactive application simulates vectorial capacity in vector-borne disease transmission using Macdonald's model.
    Adjust the parameters using the sliders and see how they affect the vectorial capacity and disease transmission potential.
    
    **Definition**: Vectorial capacity is a measure of the transmission potential of a vector population, defined as the average number 
    of potentially infective bites that would eventually arise from all the vectors that bite a single infectious host on a single day.
    
    **Core Concept**: At the heart of Macdonald's model is the calculation of how efficiently a vector population can transmit a pathogen 
    based on vector density, biting behavior, vector competence, survival rate, and pathogen development time within the vector.
    
    **Macdonald's Equation**: 
    $V = \\frac{ma^2p^n}{-\\ln(p)}b$ 
    
    Where:
    - $m$ = vector density in relation to hosts
    - $a$ = biting rate (humans bitten per mosquito per day)
    - $p$ = daily survival rate (especially of adults)
    - $n$ = extrinsic incubation period (days from ingestion to transmission)
    - $b$ = vector competence (proportion of vectors that develop infection)
    
    **The Model's Logic**:
    1. Vector density (m) directly affects how many vectors are available to transmit disease
    2. Biting rate (a) appears squared because vectors must bite twice to transmit (acquire and deliver)
    3. Survival (p) is critical since vectors must live long enough to become infectious (p^n)
    4. Vector lifespan (1/-ln(p)) determines how many infectious bites a vector can deliver
    5. Vector competence (b) determines what proportion of vectors can actually transmit
    """)
    
    # Create sidebar with parameters
    st.sidebar.header("Model Parameters")
    
    # Vector density parameters
    vector_host_ratio = st.sidebar.slider("Vector:host ratio (m)", 0.1, 50.0, 10.0, 0.1, 
                                        help="Number of vectors per human host")
    
    # Biting behavior parameters
    biting_rate = st.sidebar.slider("Daily biting rate (a)", 0.0, 1.0, 0.3, 0.01, 
                                 help="Probability that a vector feeds on a host in one day")
    
    # Host preference calculation
    st.sidebar.subheader("Host Preference Factors")
    host_preference = st.sidebar.slider("Host preference index (0-1)", 0.0, 1.0, 0.5, 0.01,
                                     help="Preference for human hosts vs other animals (1 = feeds only on humans)")
    
    days_between_feedings = st.sidebar.number_input("Number of days between feedings", 1, 14, 3, 
                                                help="Average number of days between blood meals")
    
    # Vector competence parameter
    vector_competence = st.sidebar.slider("Vector competence (b)", 0.0, 1.0, 0.5, 0.01,
                                       help="Proportion of vectors that develop infection after feeding on an infectious host")
    
    # Survival parameters
    daily_survival = st.sidebar.slider("Daily survival probability (p)", 0.5, 0.99, 0.8, 0.01,
                                    help="Probability that a vector survives one day")
    
    # Calculate vector lifespan after EIP
    vector_lifespan = 1/(-math.log(daily_survival)) if daily_survival > 0 else 0
    st.sidebar.text(f"Vector lifespan: {vector_lifespan:.2f} days")
    
    # Pathogen parameters
    extrinsic_incubation = st.sidebar.slider("Extrinsic incubation period (n)", 1, 30, 10, 1,
                                          help="Days required for pathogen development in vector")
    
    # Advanced features
    st.sidebar.subheader("Advanced Model Features")
    enable_seasonality = st.sidebar.checkbox("Enable Seasonal Fluctuations", False, 
                                          help="Simulate seasonal changes in vector density")
    if enable_seasonality:
        seasonality_amplitude = st.sidebar.slider("Seasonality Amplitude (%)", 0, 100, 50, 
                                               help="Maximum percent change in vector density due to seasonality")
    else:
        seasonality_amplitude = 0
    
    # Function to calculate vectorial capacity
    def calculate_vectorial_capacity(m, a, b, p, n):
        """
        Calculate the vectorial capacity using Macdonald's formula
        
        Parameters:
        - m: Vector:host ratio
        - a: Daily biting rate
        - b: Vector competence
        - p: Daily survival probability
        - n: Extrinsic incubation period
        
        Returns:
        - V: Vectorial capacity
        """
        # Prevent domain errors with invalid parameters
        if p <= 0 or p >= 1:
            return 0
        
        # Handle other potential errors
        try:
            # Macdonald's formula: V = ma²bp^n/-ln(p)
            return (m * (a**2) * b * (p**n)) / (-math.log(p))
        except Exception:
            # Return 0 if any calculation error occurs
            return 0
    
    # Calculate vectorial capacity with current parameters
    vectorial_capacity = calculate_vectorial_capacity(
        vector_host_ratio,
        biting_rate,
        vector_competence,
        daily_survival,
        extrinsic_incubation
    )
    
    # Calculate basic reproduction number (R0) assuming human recovery rate of 0.14 (~ 7 days infectious period)
    recovery_rate = 0.14  # recovery rate per day
    r0 = vectorial_capacity * vector_competence / recovery_rate
    
    # Calculate minimum survival rate needed to maintain transmission
    if biting_rate > 0 and vector_competence > 0 and vector_host_ratio > 0:
        critical_daily_survival = math.exp(-1/extrinsic_incubation)
    else:
        critical_daily_survival = 0
    
    # Display the overall statistics
    st.header("Vectorial Capacity Results")
    
    # Try to use plotly for the gauge if available
    try:
        if PLOTLY_AVAILABLE:
            # Create two columns - gauge on left, metrics on right
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create a gauge visualization for vectorial capacity
                # Define a reference value (0.587) from Table 24.1 in the reading
                reference_value = 0.587
                
                # Calculate percent change
                percent_change = ((vectorial_capacity - reference_value) / reference_value) * 100 if reference_value > 0 else 0
                
                # Create a gauge figure
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=vectorial_capacity,
                    delta={'reference': reference_value, 'relative': False, 'valueformat': '.3f'},
                    title={'text': "Vectorial Capacity", 'font': {'size': 24}},
                    gauge={
                        'axis': {'range': [None, max(5, vectorial_capacity * 1.5)]},
                        'bar': {'color': "darkblue"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 0.5], 'color': "lightgreen"},
                            {'range': [0.5, 1], 'color': "yellow"},
                            {'range': [1, max(5, vectorial_capacity * 1.5)], 'color': "salmon"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': reference_value
                        }
                    }
                ))
                
                fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add caption for the gauge
                st.caption(f"Reference baseline value (red line): {reference_value}")
                st.caption(f"Current value: {vectorial_capacity:.3f} ({percent_change:+.1f}%)")
            
            with col2:
                # Display metrics in vertical stack
                st.metric("Vectorial Capacity (V)", f"{vectorial_capacity:.4f}")
                st.metric("Basic Reproduction Number (R₀)", f"{r0:.4f}", 
                       help="R₀ = Vectorial Capacity × Vector Competence ÷ Human Recovery Rate")
                st.metric("Critical Daily Survival", f"{critical_daily_survival:.4f}",
                       delta=f"{(daily_survival - critical_daily_survival):.4f}", 
                       delta_color="normal",
                       help="Minimum daily survival rate needed for vectors to live long enough to transmit the pathogen")
        else:
            raise ImportError("Plotly not available")
    except:
        # Fallback to standard metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Vectorial Capacity (V)", f"{vectorial_capacity:.4f}")
            reference_value = 0.587
            percent_change = ((vectorial_capacity - reference_value) / reference_value) * 100 if reference_value > 0 else 0
            st.caption(f"Change from reference: {percent_change:+.1f}%")
            
        with col2:
            st.metric("Basic Reproduction Number (R₀)", f"{r0:.4f}", 
                   help="R₀ = Vectorial Capacity × Vector Competence ÷ Human Recovery Rate")
            
        with col3:
            st.metric("Critical Daily Survival", f"{critical_daily_survival:.4f}",
                   delta=f"{(daily_survival - critical_daily_survival):.4f}", 
                   delta_color="normal",
                   help="Minimum daily survival rate needed for vectors to live long enough to transmit the pathogen")
    
    # Helper function to convert figure to downloadable data
    def fig_to_bytes(fig):
        """Convert a matplotlib figure to bytes for downloading"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        return buf
    
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Vectorial Capacity", "Sensitivity Analysis", "Host Preference", "Parameter Relationships", "Vectorial Parameters"])
    
    with tab1:
        st.header("Parameter Impact on Vectorial Capacity")
        
        # Calculate the impact of changing each parameter by ±10%
        impact_data = {}
        params = {
            "Vector:host ratio (m)": {"value": vector_host_ratio, "code": "m"},
            "Biting rate (a)": {"value": biting_rate, "code": "a"},
            "Vector competence (b)": {"value": vector_competence, "code": "b"},
            "Daily survival (p)": {"value": daily_survival, "code": "p"},
            "EIP (n)": {"value": extrinsic_incubation, "code": "n"}
        }
        
        for param_name, param_info in params.items():
            # Get baseline values
            baseline_params = {
                "m": vector_host_ratio,
                "a": biting_rate,
                "b": vector_competence,
                "p": daily_survival,
                "n": extrinsic_incubation
            }
            
            # Calculate baseline VC
            baseline_vc = calculate_vectorial_capacity(
                baseline_params["m"], baseline_params["a"], 
                baseline_params["b"], baseline_params["p"], 
                baseline_params["n"]
            )
            
            # Calculate VC with +10% parameter change
            increased_params = baseline_params.copy()
            increased_params[param_info["code"]] *= 1.1
            increased_vc = calculate_vectorial_capacity(
                increased_params["m"], increased_params["a"], 
                increased_params["b"], increased_params["p"], 
                increased_params["n"]
            )
            
            # Calculate VC with -10% parameter change
            decreased_params = baseline_params.copy()
            decreased_params[param_info["code"]] *= 0.9
            decreased_vc = calculate_vectorial_capacity(
                decreased_params["m"], decreased_params["a"], 
                decreased_params["b"], decreased_params["p"], 
                decreased_params["n"]
            )
            
            # Calculate percent changes
            pct_change_up = ((increased_vc - baseline_vc) / baseline_vc) * 100 if baseline_vc > 0 else 0
            pct_change_down = ((decreased_vc - baseline_vc) / baseline_vc) * 100 if baseline_vc > 0 else 0
            
            # Store results - using plain numbers instead of formatted strings
            impact_data[param_name] = {
                "Current Value": param_info["value"],
                "+10% Change": pct_change_up,  # Store as number, not string
                "-10% Change": pct_change_down  # Store as number, not string
            }
        
        # Create a DataFrame
        impact_df = pd.DataFrame(impact_data).T.reset_index()
        impact_df.columns = ["Parameter", "Current Value", "+10% Parameter → %ΔV", "-10% Parameter → %ΔV"]
        
        # Display the table
        st.dataframe(impact_df)
        
        # Create bar chart to visualize parameter impact
        try:
            # Only use plotly if it's available
            if PLOTLY_AVAILABLE:
                fig_impact = go.Figure()
                
                # Extract numeric values
                plus_values = impact_df["+10% Parameter → %ΔV"].tolist()
                minus_values = impact_df["-10% Parameter → %ΔV"].tolist()
                
                # Add bars for +10% impact
                fig_impact.add_trace(go.Bar(
                    x=impact_df["Parameter"],
                    y=plus_values,
                    name="+10% Parameter Change",
                    marker_color='forestgreen'
                ))
                
                # Add bars for -10% impact
                fig_impact.add_trace(go.Bar(
                    x=impact_df["Parameter"],
                    y=minus_values,
                    name="-10% Parameter Change",
                    marker_color='crimson'
                ))
                
                # Update layout
                fig_impact.update_layout(
                    title="Impact of ±10% Parameter Changes on Vectorial Capacity",
                    xaxis_title="Parameter",
                    yaxis_title="% Change in Vectorial Capacity",
                    barmode='group',
                    height=500
                )
                
                st.plotly_chart(fig_impact, use_container_width=True)
            else:
                # Fallback to matplotlib if plotly is not available
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Extract numeric values - already numbers
                plus_values = impact_df["+10% Parameter → %ΔV"].tolist()
                minus_values = impact_df["-10% Parameter → %ΔV"].tolist()
                
                # Set width and positions for bars
                x = np.arange(len(impact_df["Parameter"]))
                width = 0.35
                
                # Create bars
                ax.bar(x - width/2, plus_values, width, label='+10% Parameter Change', color='forestgreen')
                ax.bar(x + width/2, minus_values, width, label='-10% Parameter Change', color='crimson')
                
                # Add labels and legend
                ax.set_xlabel('Parameter')
                ax.set_ylabel('% Change in Vectorial Capacity')
                ax.set_title('Impact of ±10% Parameter Changes on Vectorial Capacity')
                ax.set_xticks(x)
                ax.set_xticklabels(impact_df["Parameter"], rotation=45, ha='right')
                ax.legend()
                
                fig.tight_layout()
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not create bar chart visualization: {e}")
            st.write("Parameter impact is shown in the table above.")
        
        # Explanation of parameter impacts based on the reading
        st.subheader("Key Parameter Effects")
        st.markdown("""
        **Parameter Impact Analysis:**
        
        As shown in Table 24.1 from the reading, three parameters most strongly affect vectorial capacity:
        
        - **Daily survivorship (p)** has the most dramatic effect on vectorial capacity
          (a 10% increase can lead to 240% increase in vectorial capacity as seen in the reading)
        
        - **Biting rate (a)** has a squared relationship with vectorial capacity
          (a 10% increase results in approximately 21% increase in vectorial capacity)
        
        - **Vector:host ratio (m)** has a linear relationship
          (a 10% increase results in approximately 10% increase in vectorial capacity)
        
        These relationships are critical for understanding which interventions might be most effective for 
        vector control. Macdonald's analysis led to the important prediction that adulticides (targeting 
        adult vector survival) would be more effective than larvicides in reducing malaria transmission.
        """)
    
    with tab2:
        st.header("Sensitivity Analysis")
        
        # Choose parameter to vary
        param_to_vary = st.selectbox(
            "Select parameter to vary:",
            ["Vector:host ratio (m)", "Biting rate (a)", "Vector competence (b)", 
             "Daily survival (p)", "Extrinsic incubation period (n)"]
        )
        
        # Set up parameter ranges
        if param_to_vary == "Vector:host ratio (m)":
            param_range = np.linspace(0.1, 50, 100)
            param_name = "m"
            x_label = "Vector:host ratio"
        elif param_to_vary == "Biting rate (a)":
            param_range = np.linspace(0.01, 1, 100)
            param_name = "a"
            x_label = "Biting rate (bites per vector per day)"
        elif param_to_vary == "Vector competence (b)":
            param_range = np.linspace(0.01, 1, 100)
            param_name = "b"
            x_label = "Vector competence (proportion)"
        elif param_to_vary == "Daily survival (p)":
            param_range = np.linspace(0.5, 0.99, 100)
            param_name = "p"
            x_label = "Daily survival probability"
        else:  # Extrinsic incubation period
            param_range = np.arange(1, 30)
            param_name = "n"
            x_label = "Extrinsic incubation period (days)"
        
        # Calculate vectorial capacity for each parameter value
        vc_values = []
        for param_val in param_range:
            if param_name == "m":
                vc = calculate_vectorial_capacity(param_val, biting_rate, vector_competence, daily_survival, extrinsic_incubation)
            elif param_name == "a":
                vc = calculate_vectorial_capacity(vector_host_ratio, param_val, vector_competence, daily_survival, extrinsic_incubation)
            elif param_name == "b":
                vc = calculate_vectorial_capacity(vector_host_ratio, biting_rate, param_val, daily_survival, extrinsic_incubation)
            elif param_name == "p":
                vc = calculate_vectorial_capacity(vector_host_ratio, biting_rate, vector_competence, param_val, extrinsic_incubation)
            else:  # n
                vc = calculate_vectorial_capacity(vector_host_ratio, biting_rate, vector_competence, daily_survival, param_val)
            vc_values.append(vc)
        
        # Create plot
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(param_range, vc_values, 'b-', linewidth=2)
        
        # Add vertical line for current parameter value
        if param_name == "m":
            current_val = vector_host_ratio
        elif param_name == "a":
            current_val = biting_rate
        elif param_name == "b":
            current_val = vector_competence
        elif param_name == "p":
            current_val = daily_survival
        else:  # n
            current_val = extrinsic_incubation
        
        # Find the current VC value
        current_vc = calculate_vectorial_capacity(
            vector_host_ratio, biting_rate, vector_competence, daily_survival, extrinsic_incubation
        )
        
        # Add vertical line for current parameter value if within range
        if min(param_range) <= current_val <= max(param_range):
            ax1.axvline(x=current_val, color='r', linestyle='--', alpha=0.7)
            ax1.plot(current_val, current_vc, 'ro', markersize=8)
            ax1.annotate(f'Current value: {current_val:.2f}\nVC: {current_vc:.2f}', 
                        xy=(current_val, current_vc), xytext=(10, -30),
                        textcoords='offset points', ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Add horizontal line at R0 = 1
        critical_vc = recovery_rate / vector_competence
        ax1.axhline(y=critical_vc, color='g', linestyle='--', alpha=0.7)
        ax1.annotate(f'Critical VC for R₀=1: {critical_vc:.2f}', 
                    xy=(param_range[len(param_range)//2], critical_vc), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5))
        
        ax1.set_xlabel(x_label, fontsize=12)
        ax1.set_ylabel('Vectorial Capacity', fontsize=12)
        ax1.set_title(f'Effect of {param_to_vary} on Vectorial Capacity', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add shaded region for R0 > 1
        ax1.fill_between(param_range, critical_vc, max(vc_values)*1.1, color='red', alpha=0.1)
        ax1.fill_between(param_range, 0, critical_vc, color='green', alpha=0.1)
        
        st.pyplot(fig1)
        
        # Download button
        st.download_button(
            label="Download Sensitivity Analysis Plot",
            data=fig_to_bytes(fig1),
            file_name=f"vc_sensitivity_{param_name}.png",
            mime="image/png"
        )
        
        # Add interpretations from Table 24.1 in the reading
        st.subheader("Interpretation")
        
        if param_name == "m":
            st.write("""
            The vector:host ratio (m) has a **linear** effect on vectorial capacity. According to Table 24.1 in the reading, 
            a 10% change in m leads to a corresponding 10% change in vectorial capacity. This parameter can be targeted through 
            vector control measures that reduce mosquito population density.
            """)
        elif param_name == "a":
            st.write("""
            The biting rate (a) has a **quadratic** effect on vectorial capacity because it appears as a² in the equation. 
            This makes it one of the most influential parameters. According to Table 24.1 in the reading, a 10% increase in 
            biting rate leads to a 21% increase in vectorial capacity. This parameter can be targeted through interventions 
            like bed nets or repellents that reduce human-vector contact.
            """)
        elif param_name == "b":
            st.write("""
            Vector competence (b) has a **linear** effect on vectorial capacity. According to Table 24.1 in the reading, 
            a 10% change in b leads to a corresponding 10% change in vectorial capacity. This biological parameter represents 
            how efficiently the vector can become infected and transmit the pathogen.
            """)
        elif param_name == "p":
            st.write("""
            Daily survival probability (p) has the most **dramatic effect** on vectorial capacity. According to Table 24.1 in the reading, 
            a 10% increase in p can lead to a 240% increase in vectorial capacity, while a 10% decrease results in a 68% decrease. 
            This powerful relationship led Macdonald to predict that adulticides would be more effective than larvicides for malaria control.
            """)
        else:  # n
            st.write("""
            The extrinsic incubation period (n) has an **inverse** effect on vectorial capacity. According to Table 24.1 in the reading, 
            a 10% increase in n leads to a 14% decrease in vectorial capacity, while a 10% decrease results in a 17% increase. 
            This parameter is primarily determined by pathogen biology and environmental conditions, especially temperature.
            """)
    
    with tab3:
        st.header("Host Preference Impact")
        
        # Create range of host preference values
        pref_range = np.linspace(0.01, 1, 100)
        
        # Calculate vectorial capacity for each preference value
        # Assuming host preference affects biting rate directly
        vc_by_preference = []
        for pref in pref_range:
            # Modify biting rate based on preference
            adjusted_biting = pref / days_between_feedings
            vc = calculate_vectorial_capacity(vector_host_ratio, adjusted_biting, vector_competence, daily_survival, extrinsic_incubation)
            vc_by_preference.append(vc)
        
        # Create plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(pref_range, vc_by_preference, color='purple', linewidth=2)
        
        # Add vertical line for current host preference
        current_pref = host_preference
        # Calculate current VC with this preference
        adjusted_current_biting = current_pref / days_between_feedings
        current_pref_vc = calculate_vectorial_capacity(vector_host_ratio, adjusted_current_biting, vector_competence, daily_survival, extrinsic_incubation)
        
        ax2.axvline(x=current_pref, color='r', linestyle='--', alpha=0.7)
        ax2.plot(current_pref, current_pref_vc, 'ro', markersize=8)
        ax2.annotate(f'Current preference: {current_pref:.2f}\nVC: {current_pref_vc:.2f}', 
                    xy=(current_pref, current_pref_vc), xytext=(10, -30),
                    textcoords='offset points', ha='left', va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Add horizontal line at R0 = 1
        critical_vc = recovery_rate / vector_competence
        ax2.axhline(y=critical_vc, color='g', linestyle='--', alpha=0.7)
        ax2.annotate(f'Critical VC for R₀=1: {critical_vc:.2f}', 
                    xy=(0.5, critical_vc), xytext=(0, 10),
                    textcoords='offset points', ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.5))
        
        ax2.set_xlabel('Host Preference Index (Human Preference)', fontsize=12)
        ax2.set_ylabel('Vectorial Capacity', fontsize=12)
        ax2.set_title('Effect of Host Preference on Vectorial Capacity', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add shaded region for R0 > 1
        ax2.fill_between(pref_range, critical_vc, max(vc_by_preference)*1.1, color='red', alpha=0.1)
        ax2.fill_between(pref_range, 0, critical_vc, color='green', alpha=0.1)
        
        st.pyplot(fig2)
        
        # Download button
        st.download_button(
            label="Download Host Preference Plot",
            data=fig_to_bytes(fig2),
            file_name="vc_host_preference.png",
            mime="image/png"
        )
        
        # Add contextual info
        st.subheader("Host Preference and Vector-Borne Disease")
        
        # Calculate threshold preference for disease persistence
        threshold_found = False
        threshold_preference = 0
        
        for i, pref in enumerate(pref_range):
            if vc_by_preference[i] >= critical_vc:
                threshold_preference = pref
                threshold_found = True
                break
        
        # Format the threshold preference as string to avoid f-string issues
        if threshold_found:
            threshold_text = f"{threshold_preference:.2f}"
        else:
            threshold_text = "not reached with these parameters"
            
        st.write(f"""
        Host preference is a crucial factor in vector-borne disease transmission. As mentioned in the reading, it determines how frequently vectors feed on humans versus other animals.
        
        **Anthropophilic vectors** (those that prefer human hosts) generally have higher vectorial capacity for human diseases. 
        **Zoophilic vectors** (those that prefer animal hosts) may serve as less efficient vectors for human pathogens.
        
        Based on the current parameters, the minimum host preference index needed for sustained transmission (R₀>1) is approximately {threshold_text}.
        
        **Implications for control:**
        - Zooprophylaxis: Using animal hosts as "bait" to divert vectors from humans
        - Targeted interventions based on vector feeding behavior
        - Community-level protection strategies accounting for vector preference
        """)
    
    with tab4:
        st.header("Parameter Relationships")
        
        # Select parameters for X and Y axes
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox(
                "Select parameter for X-axis:",
                ["Vector:host ratio (m)", "Biting rate (a)", "Vector competence (b)", 
                 "Daily survival (p)", "Extrinsic incubation period (n)"],
                index=0
            )
        
        with col2:
            y_param = st.selectbox(
                "Select parameter for Y-axis:",
                ["Vector:host ratio (m)", "Biting rate (a)", "Vector competence (b)", 
                 "Daily survival (p)", "Extrinsic incubation period (n)"],
                index=3
            )
        
        # Ensure different parameters are selected
        if x_param == y_param:
            st.error("Please select different parameters for X and Y axes.")
        else:
            # Set up parameter ranges
            param_configs = {
                "Vector:host ratio (m)": {
                    "range": np.linspace(0.1, 50, 20),
                    "name": "m",
                    "label": "Vector:host ratio"
                },
                "Biting rate (a)": {
                    "range": np.linspace(0.05, 1, 20),
                    "name": "a",
                    "label": "Biting rate (bites per day)"
                },
                "Vector competence (b)": {
                    "range": np.linspace(0.05, 1, 20),
                    "name": "b",
                    "label": "Vector competence"
                },
                "Daily survival (p)": {
                    "range": np.linspace(0.7, 0.99, 20),
                    "name": "p",
                    "label": "Daily survival probability"
                },
                "Extrinsic incubation period (n)": {
                    "range": np.arange(1, 21),
                    "name": "n",
                    "label": "Extrinsic incubation period (days)"
                }
            }
            
            x_range = param_configs[x_param]["range"]
            y_range = param_configs[y_param]["range"]
            x_name = param_configs[x_param]["name"]
            y_name = param_configs[y_param]["name"]
            
            # Create meshgrid
            X, Y = np.meshgrid(x_range, y_range)
            Z = np.zeros_like(X)
            
            # Calculate vectorial capacity for each combination
            for i in range(len(y_range)):
                for j in range(len(x_range)):
                    try:
                        # Get parameter values
                        params = {
                            "m": vector_host_ratio,
                            "a": biting_rate,
                            "b": vector_competence,
                            "p": daily_survival,
                            "n": extrinsic_incubation
                        }
                        
                        # Update with grid values
                        params[x_name] = X[i, j]
                        params[y_name] = Y[i, j]
                        
                        # Calculate VC
                        Z[i, j] = calculate_vectorial_capacity(
                            params["m"], params["a"], params["b"], params["p"], params["n"]
                        )
                    except Exception:
                        # Handle any errors during calculation
                        Z[i, j] = 0
            
            # Create heatmap
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            contour = ax3.contourf(X, Y, Z, 20, cmap='viridis')
            
            # Add contour lines for R0 = 1
            critical_vc = recovery_rate / vector_competence
            critical_contour = ax3.contour(X, Y, Z, levels=[critical_vc], colors='red', linestyles='dashed', linewidths=2)
            ax3.clabel(critical_contour, inline=True, fontsize=10, fmt=f'R₀=1 (VC={critical_vc:.2f})')
            
            # Mark current parameter values
            current_x = params[x_name]
            current_y = params[y_name]
            
            # Check if current values are within the plotted range
            if (min(x_range) <= current_x <= max(x_range)) and (min(y_range) <= current_y <= max(y_range)):
                ax3.plot(current_x, current_y, 'ro', markersize=10)
                ax3.annotate('Current values', xy=(current_x, current_y), xytext=(10, 10),
                            textcoords='offset points', ha='left', va='bottom',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax3)
            cbar.set_label('Vectorial Capacity', fontsize=12)
            
            ax3.set_xlabel(param_configs[x_param]["label"], fontsize=12)
            ax3.set_ylabel(param_configs[y_param]["label"], fontsize=12)
            ax3.set_title(f'Vectorial Capacity as a Function of {x_param} and {y_param}', fontsize=14)
            
            st.pyplot(fig3)
            
            # Download button
            st.download_button(
                label="Download Parameter Relationship Plot",
                data=fig_to_bytes(fig3),
                file_name=f"vc_relationship_{x_name}_{y_name}.png",
                mime="image/png"
            )
            
            # Explanation
            st.subheader("Parameter Interaction Effects")
            st.write(f"""
            This heatmap shows how vectorial capacity changes as a function of both {x_param.lower()} and {y_param.lower()}.
            
            **Key observations:**
            - The red dashed line represents the critical threshold for disease persistence (R₀=1)
            - Areas with higher values (yellow/green) indicate conditions more favorable for disease transmission
            - Areas with lower values (blue/purple) indicate conditions less favorable for transmission
            
            **Control implications:**
            - Understanding parameter interactions helps identify the most effective combination of interventions
            - Some parameter combinations may have synergistic effects on reducing vectorial capacity
            - The steepness of the gradient indicates sensitivity to parameter changes
            """)
    
    with tab5:
        st.header("Life Table and Calculations")
        
        # Create a life table for mosquitoes based on the reading
        st.subheader("Vector Life Table")
        
        # Create a life table based on Table 24.2 from the reading
        life_table_data = {
            "Life stage": ["Egg", "Larva", "Larva", "Larva", "Larva", "Larva", "Larva", "Pupa", 
                          "Adult", "Adult", "Adult", "Adult", "Adult", "Adult", "Adult", "Adult",
                          "Adult", "Adult", "Adult", "Adult", "Adult", "Adult", "Adult", "Adult",
                          "Adult", "Adult", "Adult", "Adult", "Adult"],
            "Chronological age (days)": list(range(29)),
            "Probability of daily survival (p)": [0.5] + [0.6]*7 + [0.9]*21,
            "Fecundity (f)": [0]*12 + [120] + [0]*4 + [100] + [0]*4 + [80] + [0]*4 + [60] + [0]
        }
        
        life_table_df = pd.DataFrame(life_table_data)
        
        # Display the life table
        st.dataframe(life_table_df)
        
        st.write("""
        This life table is based on Table 24.2 from the reading. It shows:
        
        - **Egg survivorship** is 0.5 (50% daily survival)
        - **Larval survivorship** is 0.6 (60% daily survival)
        - **Adult survivorship** is 0.9 (90% daily survival)
        - **Fecundity** occurs on days 12, 17, 22, and 27, with decreasing egg output each time
        
        The life table approach is essential for understanding vectorial capacity, as it incorporates both survivorship and reproduction patterns.
        """)
        
        # Create a table showing current parameter values and calculated results
        st.subheader("Input Parameters")
        params_df = pd.DataFrame({
            'Parameter': ['Vector:host ratio (m)', 'Biting rate (a)', 'Host preference index', 
                         'Days between feedings', 'Vector competence (b)', 'Daily survival (p)', 
                         'Extrinsic incubation period (n)'],
            'Value': [vector_host_ratio, biting_rate, host_preference, 
                     days_between_feedings, vector_competence, daily_survival, 
                     extrinsic_incubation],
            'Description': ['Number of vectors per host', 'Probability that a vector feeds on a host in one day', 
                           'Preference for human hosts vs other animals', 'Average number of days between blood meals',
                           'Proportion of vectors that become infective', 'Probability a vector survives one day',
                           'Days required for pathogen development in vector']
        })
        
        st.dataframe(params_df)
        
        # Derived parameters
        st.subheader("Derived Parameters")
        derived_df = pd.DataFrame({
            'Parameter': ['Vector lifespan (1/-ln(p))', 'Probability of surviving EIP (p^n)', 
                         'Vectorial Capacity (V)', 'Basic Reproduction Number (R₀)'],
            'Value': [vector_lifespan, daily_survival**extrinsic_incubation, 
                     vectorial_capacity, r0],
            'Description': ['Average vector lifespan in days', 'Probability a vector survives the entire EIP',
                           'Average number of potentially infective bites from vectors that bit one host on one day',
                           'Average number of secondary human cases arising from one human case']
        })
        
        st.dataframe(derived_df)
        
        # Create sensitivity indices table
        st.subheader("Parameter Sensitivity")
        
        # Calculate elasticity (proportional sensitivity) for each parameter
        def calculate_elasticity(param_name, base_value, delta=0.01):
            """Calculate elasticity (proportional sensitivity) for a parameter"""
            try:
                params = {
                    "m": vector_host_ratio,
                    "a": biting_rate,
                    "b": vector_competence,
                    "p": daily_survival,
                    "n": extrinsic_incubation
                }
                
                # Base VC value
                base_vc = calculate_vectorial_capacity(
                    params["m"], params["a"], params["b"], params["p"], params["n"]
                )
                
                # Handle division by zero
                if base_vc == 0:
                    return 0
                
                # Perturbed value (increase by delta %)
                # Make a copy to avoid modifying original values
                perturbed_params = params.copy()
                perturbed_params[param_name] *= (1 + delta)
                
                # New VC value
                new_vc = calculate_vectorial_capacity(
                    perturbed_params["m"], perturbed_params["a"], 
                    perturbed_params["b"], perturbed_params["p"], 
                    perturbed_params["n"]
                )
                
                # Calculate elasticity
                percent_change_vc = (new_vc - base_vc) / base_vc
                elasticity = percent_change_vc / delta
                
                return elasticity
            except Exception:
                # Return 0 if any calculation error occurs
                return 0
        
        # Calculate elasticities
        elasticities = {
            "m": calculate_elasticity("m", vector_host_ratio),
            "a": calculate_elasticity("a", biting_rate),
            "b": calculate_elasticity("b", vector_competence),
            "p": calculate_elasticity("p", daily_survival),
            "n": calculate_elasticity("n", extrinsic_incubation)
        }
        
        # Create sensitivity table
        sensitivity_df = pd.DataFrame({
            'Parameter': ['Vector:host ratio (m)', 'Biting rate (a)', 'Vector competence (b)', 
                         'Daily survival (p)', 'Extrinsic incubation period (n)'],
            'Elasticity': [elasticities["m"], elasticities["a"], elasticities["b"], 
                          elasticities["p"], elasticities["n"]],
            'Interpretation': [
                'Linear effect (elasticity ≈ 1)',
                'Quadratic effect (elasticity ≈ 2)',
                'Linear effect (elasticity ≈ 1)',
                f'Complex effect (depends on parameter value)',
                f'Exponential, inverse effect through p^n'
            ]
        })
        
        st.dataframe(sensitivity_df)
        
        # Add download buttons for CSV
        csv = params_df.to_csv(index=False)
        st.download_button(
            label="Download Parameters as CSV",
            data=csv,
            file_name="vectorial_capacity_parameters.csv",
            mime="text/csv"
        )
        
        derived_csv = derived_df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=derived_csv,
            file_name="vectorial_capacity_results.csv",
            mime="text/csv"
        )
        
        # Add conclusions section
        st.header("Connecting Vectorial Capacity to Disease Control")
        
        st.markdown("""
        **Key Insights from Macdonald's Model:**
        
        1. **Parameter Importance**: 
           - Daily survivorship (p) is the most sensitive parameter
           - This led Macdonald to predict that adulticides would be more effective than larvicides
           - Small changes in adult survival have substantial effects on disease transmission
        
        2. **Practical Applications**:
           - This model provides a framework for comparing different control strategies
           - Understanding the factors affecting vectorial capacity helps target interventions effectively
           - The original application was for malaria control, but the principles apply to many vector-borne diseases
        
        3. **Model Limitations**:
           - As noted in the reading, these models are didactic tools with simplified assumptions
           - Real vector populations show fluctuations in density and age structure
           - The comparative approach is more valuable than absolute predictions
           
        The Macdonald model continues to be a foundational framework for understanding vector-borne disease transmission
        and developing effective control strategies.
        """)

if __name__ == "__main__":
    run()
