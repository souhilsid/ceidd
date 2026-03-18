# core/visualization.py - FIXED VERSION
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st

from .optimization import OptimizationResult
from .evaluators import GeneralizedEvaluator

class VisualizationEngine:
    """Generalized visualization engine for optimization results"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_optimization_progress(self, result: OptimizationResult) -> go.Figure:
        """Plot optimization progress over batches"""
        if not result.history:
            return self._create_empty_plot("No optimization data available")
        
        df_history = pd.DataFrame(result.history)
        
        fig = go.Figure()
        
        # Best overall distance
        fig.add_trace(go.Scatter(
            x=df_history['batch'],
            y=df_history['best_overall_distance'],
            mode='lines+markers',
            name='Best Overall',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # Batch min distance
        fig.add_trace(go.Scatter(
            x=df_history['batch'],
            y=df_history['batch_min_distance'],
            mode='lines+markers',
            name='Batch Best',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Batch mean distance
        fig.add_trace(go.Scatter(
            x=df_history['batch'],
            y=df_history['batch_mean_distance'],
            mode='lines+markers',
            name='Batch Mean',
            line=dict(color='green', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Optimization Progress",
            xaxis_title="Batch",
            yaxis_title="Distance",
            hovermode='x unified',
            showlegend=True
        )
        
        return fig

    def plot_pareto_front(self, pareto_front: List[Dict[str, Any]],
                        objective_configs: List[Any]) -> go.Figure:
        """Plot Pareto front for direct objective mode"""
        if not pareto_front:
            return self._create_empty_plot("Pareto front not available yet")
        
        rows = []
        for point in pareto_front:
            row = {}
            row['parameters'] = str(point.get('parameters', {}))
            outcomes = point.get('outcomes', {})
            for name, value in outcomes.items():
                row[name] = value
            rows.append(row)
        
        df = pd.DataFrame(rows)
        outcome_cols = [c for c in df.columns if c != 'parameters']
        if len(outcome_cols) < 2:
            return self._create_empty_plot("Pareto front needs at least two objectives to plot")
        
        x_col, y_col = outcome_cols[:2]
        fig = px.scatter(
            df, x=x_col, y=y_col,
            hover_data=['parameters'],
            title="Pareto Front (direct objectives)",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        # Show only points (no connecting line) with clearer styling
        fig.update_traces(
            mode='markers',
            marker=dict(size=10, line=dict(width=1, color='rgba(0,0,0,0.35)'))
        )
        fig.update_layout(
            xaxis_title=x_col,
            yaxis_title=y_col,
            legend_title="Objectives"
        )
        return fig
    
    def plot_model_performance(self, evaluator: GeneralizedEvaluator) -> go.Figure:
        """Plot model performance comparison"""
        if not evaluator.metrics:
            return self._create_empty_plot("No model metrics available")
        
        # Prepare data for plotting
        performance_data = []
        for model_name, metrics in evaluator.metrics.items():
            for obj_name, obj_metrics in metrics.items():
                performance_data.append({
                    'Model': model_name,
                    'Objective': obj_name,
                    'R² Score': obj_metrics.test_r2,
                    'RMSE': obj_metrics.test_rmse,
                    'Quality Score': obj_metrics.quality_score,
                    'Meets Constraints': 'Yes' if obj_metrics.meets_constraints else 'No'
                })
        
        df_performance = pd.DataFrame(performance_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['R² Scores by Model', 'RMSE by Model', 
                          'Quality Scores by Model', 'Constraints Met'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # R² Scores
        r2_fig = px.bar(df_performance, x='Model', y='R² Score', color='Objective',
                       barmode='group')
        for trace in r2_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # RMSE
        rmse_fig = px.bar(df_performance, x='Model', y='RMSE', color='Objective',
                         barmode='group')
        for trace in rmse_fig.data:
            fig.add_trace(trace, row=1, col=2)
        
        # Quality Scores
        quality_fig = px.bar(df_performance, x='Model', y='Quality Score', color='Objective',
                           barmode='group')
        for trace in quality_fig.data:
            fig.add_trace(trace, row=2, col=1)
        
        # Constraints met
        constraints_data = df_performance.groupby(['Model', 'Meets Constraints']).size().reset_index(name='Count')
        constraints_fig = px.bar(constraints_data, x='Model', y='Count', color='Meets Constraints',
                               barmode='stack')
        for trace in constraints_fig.data:
            fig.add_trace(trace, row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False, title_text="Model Performance Comparison")
        return fig
    
    def plot_parallel_coordinates(self, result: OptimizationResult, 
                                parameter_configs: List[Any], 
                                objective_configs: List[Any],
                                show_distance: bool = True) -> go.Figure:
        """Plot parallel coordinates for candidates - FIXED VERSION"""
        if not result.all_candidates:
            return self._create_empty_plot("No candidate data available")
        
        # Prepare data for parallel coordinates
        data = []
        objective_names = [obj.name for obj in objective_configs]
        
        for candidate in result.all_candidates[:50]:  # Limit to first 50 for clarity
            row = {}
            
            # Add parameters
            for param_name in candidate['parameters']:
                row[param_name] = candidate['parameters'][param_name]
            
            # Add objectives - FIXED: Handle different prediction formats
            predictions = candidate['predictions']
            
            # Handle different prediction formats
            if isinstance(predictions, list):
                # List format - use directly
                pred_array = predictions
            elif isinstance(predictions, np.ndarray):
                # Numpy array - convert to list
                pred_array = predictions.tolist()
            else:
                # Unknown format - skip this candidate
                continue
            
            # Ensure we have enough predictions for all objectives
            if len(pred_array) >= len(objective_names):
                for i, obj_name in enumerate(objective_names):
                    if i < len(pred_array):
                        row[obj_name] = pred_array[i]
                    else:
                        row[obj_name] = 0.0  # Default value if missing
            else:
                # Not enough predictions, skip this candidate
                continue
            
            if show_distance and 'distance' in candidate:
                row['distance'] = candidate['distance']
                row['is_best'] = candidate['distance'] == result.best_distance
            
            data.append(row)
        
        if not data:
            return self._create_empty_plot("No valid candidate data available")
        
        df_candidates = pd.DataFrame(data)
        
        # Create dimensions for parallel coordinates
        dimensions = []
        
        # Add parameter dimensions
        for param_config in parameter_configs:
            if param_config.name in df_candidates.columns:
                dimensions.append(
                    dict(range=[df_candidates[param_config.name].min(), 
                               df_candidates[param_config.name].max()],
                         label=param_config.name, values=df_candidates[param_config.name])
                )
        
        # Add objective dimensions
        for obj_config in objective_configs:
            if obj_config.name in df_candidates.columns:
                dimensions.append(
                    dict(range=[df_candidates[obj_config.name].min(), 
                               df_candidates[obj_config.name].max()],
                         label=obj_config.name, values=df_candidates[obj_config.name])
                )
        
        # Add distance dimension
        if show_distance and 'distance' in df_candidates.columns:
            dimensions.append(
                dict(range=[df_candidates['distance'].min(), df_candidates['distance'].max()],
                     label='Distance', values=df_candidates['distance'])
            )
        
        if not dimensions:
            return self._create_empty_plot("No valid dimensions for parallel coordinates")
        
        line_kwargs = {}
        if show_distance and 'distance' in df_candidates.columns:
            line_kwargs = dict(color=df_candidates['distance'],
                         colorscale='Viridis',
                         showscale=True,
                         cmin=df_candidates['distance'].min(),
                         cmax=df_candidates['distance'].max())
        else:
            color_column = objective_names[0] if objective_names else None
            if color_column and color_column in df_candidates.columns:
                line_kwargs = dict(color=df_candidates[color_column],
                                   colorscale='Blues',
                                   showscale=False)
            else:
                line_kwargs = dict(color='blue')
        
        fig = go.Figure(data=
            go.Parcoords(
                line=line_kwargs,
                dimensions=dimensions
            )
        )
        
        fig.update_layout(title="Parallel Coordinates Plot of Candidates")
        return fig
    
    def plot_prediction_vs_actual(self, evaluator: GeneralizedEvaluator, 
                                X: np.ndarray, Y: np.ndarray) -> go.Figure:
        """Plot prediction vs actual for all models and objectives"""
        n_models = len(evaluator.models)
        n_objectives = len(evaluator.objective_names)
        
        if n_models == 0 or n_objectives == 0:
            return self._create_empty_plot("No model data available")
        
        fig = make_subplots(
            rows=n_models, cols=n_objectives,
            subplot_titles=[f"{model} - {obj}" for model in evaluator.models 
                          for obj in evaluator.objective_names]
        )
        
        row_idx = 1
        for model_name, model in evaluator.models.items():
            predictions = evaluator.predict(X, model_name)
            
            for col_idx, obj_name in enumerate(evaluator.objective_names, 1):
                obj_predictions = predictions[:, col_idx-1] if predictions.shape[1] > 1 else predictions.flatten()
                obj_actual = Y[:, col_idx-1]
                
                # Prediction vs Actual scatter
                fig.add_trace(
                    go.Scatter(
                        x=obj_actual, y=obj_predictions,
                        mode='markers',
                        name=f'{model_name} - {obj_name}',
                        showlegend=False
                    ),
                    row=row_idx, col=col_idx
                )
                
                # Perfect prediction line
                min_val = min(obj_actual.min(), obj_predictions.min())
                max_val = max(obj_actual.max(), obj_predictions.max())
                fig.add_trace(
                    go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines',
                        line=dict(dash='dash', color='red'),
                        showlegend=False
                    ),
                    row=row_idx, col=col_idx
                )
            
            row_idx += 1
        
        fig.update_layout(height=300 * n_models, title_text="Predictions vs Actual")
        fig.update_xaxes(title_text="Actual")
        fig.update_yaxes(title_text="Predicted")
        
        return fig
    
    def plot_candidate_evolution(self, result: OptimizationResult) -> go.Figure:
        """Plot how candidates evolve over optimization batches"""
        if not result.all_candidates:
            return self._create_empty_plot("No candidate evolution data available")
        
        # Group candidates by batch (simplified - in real implementation, track batches)
        batch_size = 5  # Assume batch size for demonstration
        batches = []
        current_batch = []
        
        for i, candidate in enumerate(result.all_candidates):
            current_batch.append(candidate)
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:
            batches.append(current_batch)
        
        # Plot evolution of best candidate in each batch
        batch_bests = []
        for i, batch in enumerate(batches):
            best_in_batch = min(batch, key=lambda x: x['distance'])
            batch_bests.append({
                'batch': i + 1,
                'distance': best_in_batch['distance'],
                'parameters': best_in_batch['parameters']
            })
        
        df_evolution = pd.DataFrame(batch_bests)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_evolution['batch'],
            y=df_evolution['distance'],
            mode='lines+markers',
            name='Best Candidate per Batch',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Add overall best
        if result.best_distance < float('inf'):
            fig.add_hline(y=result.best_distance, line_dash="dash", 
                         line_color="red", annotation_text="Overall Best")
        
        fig.update_layout(
            title="Candidate Evolution Over Batches",
            xaxis_title="Batch",
            yaxis_title="Distance",
            showlegend=True
        )
        
        return fig
    
    def plot_parameter_importance(self, result: OptimizationResult,
                                parameter_configs: List[Any],
                                distance_based: bool = True) -> go.Figure:
        """Plot parameter importance based on correlation with distance"""
        if not distance_based:
            return self._create_empty_plot("Distance-based importance disabled for direct objectives")
        if not result.all_candidates:
            return self._create_empty_plot("No data for parameter importance")
        
        # Calculate correlations between parameters and distance
        correlations = {}
        for param_config in parameter_configs:
            param_name = param_config.name
            param_values = [c['parameters'][param_name] for c in result.all_candidates]
            distances = [c['distance'] for c in result.all_candidates]
            
            correlation = np.corrcoef(param_values, distances)[0, 1]
            if not np.isnan(correlation):
                correlations[param_name] = abs(correlation)  # Use absolute value
        
        # Sort by importance
        sorted_correlations = dict(sorted(correlations.items(), 
                                        key=lambda x: x[1], reverse=True))
        
        fig = px.bar(
            x=list(sorted_correlations.keys()),
            y=list(sorted_correlations.values()),
            title="Parameter Importance (Correlation with Distance)",
            labels={'x': 'Parameter', 'y': 'Absolute Correlation'}
        )
        
        fig.update_layout(xaxis_tickangle=-45)
        return fig
    
    def plot_uncertainty_comparison(self, result: OptimizationResult,
                              evaluator: GeneralizedEvaluator) -> go.Figure:
     """Plot uncertainty comparison for top candidates - FIXED VERSION"""
     if not result.all_candidates:
        return self._create_empty_plot("No candidate data available")
    
    # Get top 10 candidates
     top_candidates = sorted(result.all_candidates, key=lambda x: x['distance'])[:10]
    
    # Prepare uncertainty data
     uncertainty_data = []
     objective_names = [obj.name for obj in evaluator.objective_configs] if hasattr(evaluator, 'objective_configs') else evaluator.objective_names
    
     for i, candidate in enumerate(top_candidates):
        if 'uncertainties' in candidate and candidate['uncertainties']:
            for obj_name, uncertainty in candidate['uncertainties'].items():
                # Safely get prediction value
                prediction_value = 0.0
                predictions = candidate['predictions']
                
                # Handle different prediction formats safely
                if isinstance(predictions, list):
                    if obj_name in objective_names:
                        obj_idx = objective_names.index(obj_name)
                        if obj_idx < len(predictions):
                            prediction_value = predictions[obj_idx]
                    elif len(predictions) > 0:
                        prediction_value = predictions[0]  # Fallback to first prediction
                elif isinstance(predictions, np.ndarray):
                    if predictions.size > 0:
                        if len(predictions.shape) == 1:
                            if obj_name in objective_names:
                                obj_idx = objective_names.index(obj_name)
                                if obj_idx < len(predictions):
                                    prediction_value = predictions[obj_idx]
                            else:
                                prediction_value = predictions[0] if len(predictions) > 0 else 0.0
                        else:
                            # 2D array - take first element
                            prediction_value = predictions.flat[0]
                
                uncertainty_data.append({
                    'Candidate': f"Candidate {i+1}",
                    'Objective': obj_name,
                    'Uncertainty': uncertainty,
                    'Distance': candidate['distance'],
                    'Prediction': prediction_value
                })
    
     if not uncertainty_data:
        return self._create_empty_plot("No uncertainty data available")
    
     df_uncertainty = pd.DataFrame(uncertainty_data)
    
    # Create bar plot
     fig = px.bar(df_uncertainty, x='Candidate', y='Uncertainty', color='Objective',
                title="Prediction Uncertainties for Top Candidates",
                barmode='group')
    
     fig.update_layout(xaxis_tickangle=-45)
     return fig
    
    def plot_uncertainty_vs_distance(self, result: OptimizationResult,
                                   evaluator: GeneralizedEvaluator) -> go.Figure:
        """Plot uncertainty vs distance relationship"""
        if not result.all_candidates:
            return self._create_empty_plot("No candidate data available")
        
        # Prepare data
        plot_data = []
        for candidate in result.all_candidates:
            if 'uncertainties' in candidate and candidate['uncertainties']:
                avg_uncertainty = np.mean(list(candidate['uncertainties'].values()))
                plot_data.append({
                    'Distance': candidate['distance'],
                    'Avg_Uncertainty': avg_uncertainty,
                    'Is_Experimental': candidate.get('is_experimental', False)
                })
        
        if not plot_data:
            return self._create_empty_plot("No uncertainty data available")
        
        df_plot = pd.DataFrame(plot_data)
        
        # Create scatter plot
        fig = px.scatter(df_plot, x='Distance', y='Avg_Uncertainty', 
                        color='Is_Experimental',
                        title="Average Uncertainty vs Distance",
                        labels={'Avg_Uncertainty': 'Average Uncertainty',
                               'Is_Experimental': 'Experimental Data'})
        
        # Add trend line
        if len(df_plot) > 1:
            z = np.polyfit(df_plot['Distance'], df_plot['Avg_Uncertainty'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df_plot['Distance'].min(), df_plot['Distance'].max(), 100)
            fig.add_trace(go.Scatter(x=x_range, y=p(x_range), 
                                   mode='lines', name='Trend Line',
                                   line=dict(dash='dash', color='gray')))
        
        return fig

    def create_optimization_dashboard(self, result: OptimizationResult,
                               evaluator: GeneralizedEvaluator,
                               parameter_configs: List[Any],
                               objective_configs: List[Any],
                               X: np.ndarray = None,
                               Y: np.ndarray = None) -> None:
     """Create comprehensive optimization dashboard in Streamlit - FIXED VERSION"""
     st.header("📊 Optimization Dashboard")
     # Determine mode
     uses_direct = getattr(result, 'uses_direct_objectives', False)

    
    # Key metrics
     col1, col2, col3, col4 = st.columns(4)
     if uses_direct:
        with col1:
            st.metric("Mode", "Direct (Pareto)")
        with col2:
            st.metric("Pareto Points", f"{len(getattr(result, 'pareto_front', []))}")
        with col3:
            st.metric("Total Candidates", f"{len(result.all_candidates)}")
        with col4:
            strategy = result.model_performance.get('generation_strategy', 'default')
            st.metric("Strategy", strategy)
     else:
        with col1:
            st.metric("Best Distance", f"{result.best_distance:.6f}")
        with col2:
            st.metric("Total Candidates", f"{len(result.all_candidates)}")
        with col3:
            st.metric("Optimization Batches", f"{len(result.history)}")
        with col4:
            strategy = result.model_performance.get('generation_strategy', 'default')
            st.metric("Strategy", strategy)

    # Tabs for different visualizations

     tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Progress", "Model Performance", "Candidates", "Parameter Analysis", 
        "Uncertainty Analysis", "Predictions", "Exploration"
    ])
    
     with tab1:
        if uses_direct:
            st.subheader("Pareto Front")
            pareto_data = getattr(result, 'pareto_front', []) or []
            if not pareto_data:
                # Fallback: recompute from candidates if backend did not populate
                candidate_df = self._candidate_dataframe(result, parameter_configs, objective_configs)
                pareto_data = self._compute_pareto_fallback(candidate_df, parameter_configs, objective_configs)
            fig_pareto = self.plot_pareto_front(pareto_data, objective_configs)
            st.plotly_chart(fig_pareto, use_container_width=True)
            st.info("Direct objective mode uses Pareto optimization; distance-based progress plots are hidden.")
        else:
            st.subheader("Optimization Progress")
            fig_progress = self.plot_optimization_progress(result)
            st.plotly_chart(fig_progress, use_container_width=True)
            
            st.subheader("Candidate Evolution")
            fig_evolution = self.plot_candidate_evolution(result)
            st.plotly_chart(fig_evolution, use_container_width=True)

     with tab2:

        st.subheader("Model Performance Comparison")
        try:
            fig_performance = self.plot_model_performance(evaluator)
            st.plotly_chart(fig_performance, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating model performance plot: {e}")
            st.info("Model performance data may not be available")
    
     with tab3:
        st.subheader("Candidate Analysis")
        try:
            fig_parallel = self.plot_parallel_coordinates(result, parameter_configs, objective_configs, show_distance=not uses_direct)
            st.plotly_chart(fig_parallel, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating parallel coordinates plot: {e}")
            st.info("This visualization requires complete candidate prediction data.")
    
     with tab4:
        st.subheader("Parameter Importance")
        try:
            fig_importance = self.plot_parameter_importance(result, parameter_configs, distance_based=not uses_direct)
            st.plotly_chart(fig_importance, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating parameter importance plot: {e}")
    
     with tab5:
        st.subheader("Uncertainty Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            try:
                fig_uncertainty = self.plot_uncertainty_comparison(result, evaluator)
                st.plotly_chart(fig_uncertainty, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating uncertainty comparison: {e}")
        
        with col2:
            if uses_direct:
                st.info("Distance-based uncertainty plot is hidden in direct objective mode.")
            else:
                try:
                    fig_uncertainty_vs_dist = self.plot_uncertainty_vs_distance(result, evaluator)
                    st.plotly_chart(fig_uncertainty_vs_dist, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating uncertainty vs distance plot: {e}")
        
        # Enhanced top candidates table with uncertainty - FIXED VERSION
        st.subheader("dY?+ Top 5 Candidates with Uncertainty")
        if uses_direct:
            pareto_points = getattr(result, 'pareto_front', [])[:5]
            if pareto_points:
                rows = []
                for i, point in enumerate(pareto_points):
                    row = {'Rank': i + 1, 'Parameters': str(point.get('parameters', {}))}
                    for name, value in point.get('outcomes', {}).items():
                        row[name] = value
                    rows.append(row)
                df_top = pd.DataFrame(rows)
                st.dataframe(df_top, use_container_width=True)
            else:
                st.info("No Pareto front available yet.")
        else:
            top_candidates = sorted(result.all_candidates, key=lambda x: x['distance'])[:5]
            
            if top_candidates:
                candidate_data = []
                objective_names = [obj.name for obj in objective_configs]
                
                for i, candidate in enumerate(top_candidates):
                    row = {
                        'Rank': i + 1, 
                        'Distance': candidate['distance'],
                        'Trial_Type': 'Experimental' if candidate.get('is_experimental') else 'Optimized'
                    }
                    
                    # Add parameters
                    for param_name, param_value in candidate['parameters'].items():
                        row[param_name] = param_value
                    
                    # Add predictions and uncertainties - FIXED: Safe prediction extraction
                    predictions = candidate['predictions']
                    
                    # Convert predictions to consistent format
                    if isinstance(predictions, list):
                        pred_list = predictions
                    elif isinstance(predictions, np.ndarray):
                        if predictions.ndim == 1:
                            pred_list = predictions.tolist()
                        else:
                            pred_list = predictions.flatten().tolist()
                    else:
                        pred_list = []
                    
                    for j, obj_name in enumerate(objective_names):
                        # Get prediction value safely
                        if j < len(pred_list):
                            pred_value = pred_list[j]
                            # Ensure it's a scalar
                            if hasattr(pred_value, '__iter__') and not isinstance(pred_value, (int, float, str)):
                                pred_value = pred_value[0] if len(pred_value) > 0 else 0.0
                            row[f'{obj_name}'] = float(pred_value)
                        else:
                            row[f'{obj_name}'] = 0.0
                        
                        # Add uncertainty if available
                        uncertainty = candidate.get('uncertainties', {}).get(obj_name)
                        if uncertainty is not None:
                            row[f'{obj_name}_uncertainty'] = uncertainty
                            row[f'{obj_name}_range'] = f"{row[f'{obj_name}'] - uncertainty:.3f} - {row[f'{obj_name}'] + uncertainty:.3f}"
                        else:
                            row[f'{obj_name}_uncertainty'] = 0.0
                            row[f'{obj_name}_range'] = "N/A"
                    
                    candidate_data.append(row)
                
                df_top = pd.DataFrame(candidate_data)
                
                # Display the table with conditional formatting
                styled_df = df_top.style.background_gradient(
                    subset=['Distance'], cmap='RdYlGn_r'
                ).format({
                    'Distance': '{:.6f}'
                })
                
                st.dataframe(styled_df, use_container_width=True)
            else:
                st.info("No candidate data available for display")

     with tab6:

        if X is not None and Y is not None:
            st.subheader("Model Predictions vs Actual")
            try:
                fig_predictions = self.plot_prediction_vs_actual(evaluator, X, Y)
                st.plotly_chart(fig_predictions, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating prediction vs actual plot: {e}")
        else:
            st.info("Upload training data to see prediction plots")

     with tab7:
        st.subheader("Exploration & Space Filling")
        df_candidates = self._candidate_dataframe(result, parameter_configs, objective_configs)
        if df_candidates.empty:
            st.info("No candidate data available yet.")
        else:
            time_color = px.colors.sequential.Blues
            if len(df_candidates) > 1:
                df_candidates['step'] = df_candidates['index']

            # Objective evolution over iterations (works for direct and target modes)
            obj_cols = [obj.name for obj in objective_configs if obj.name in df_candidates.columns]
            if obj_cols:
                melt_df = df_candidates.melt(id_vars=['index'], value_vars=obj_cols, var_name='Objective', value_name='Value')
                fig_obj = px.line(melt_df, x='index', y='Value', color='Objective', markers=True,
                                  title="Objective values over iterations")
                st.plotly_chart(fig_obj, use_container_width=True)
            else:
                st.info("Objective values not available yet for plotting.")

            # Parameter sweep vs iteration
            param_cols = [p.name for p in parameter_configs if p.name in df_candidates.columns]
            if param_cols:
                melt_params = df_candidates.melt(id_vars=['index'], value_vars=param_cols, var_name='Parameter', value_name='Value')
                fig_params = px.line(melt_params, x='index', y='Value', color='Parameter', markers=True,
                                     title="Suggested parameters over iterations")
                st.plotly_chart(fig_params, use_container_width=True)

            # Space filling scatter for first two parameters
            if len(param_cols) >= 2:
                fig_space = px.scatter(
                    df_candidates,
                    x=param_cols[0],
                    y=param_cols[1],
                    color='index',
                    color_continuous_scale='Viridis',
                    hover_data=['index'] + obj_cols,
                    title=f"Space filling of candidates ({param_cols[0]} vs {param_cols[1]})"
                )
                st.plotly_chart(fig_space, use_container_width=True)

            # Pairwise scatter matrix for up to 4 columns (params + objectives)
            cols_for_matrix = (param_cols + obj_cols)[:4]
            if len(cols_for_matrix) >= 2:
                fig_matrix = px.scatter_matrix(
                    df_candidates,
                    dimensions=cols_for_matrix,
                    color='index',
                    color_continuous_scale='Blues',
                    title="Pairwise exploration coverage"
                )
                st.plotly_chart(fig_matrix, use_container_width=True)
    
    # Best candidate details - FIXED VERSION
     if result.best_parameters:
        st.header("🎯 Best Candidate")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Parameters")
            for param_name, param_value in result.best_parameters.items():
                st.write(f"- **{param_name}**: `{param_value:.4f}`")
        
        with col2:
            st.subheader("Predictions")
            # FIXED: Handle best_predictions format consistently
            best_predictions = result.best_predictions
            objective_names = [obj.name for obj in objective_configs]
            
            # Convert to proper format
            if isinstance(best_predictions, np.ndarray):
                if best_predictions.ndim == 1:
                    pred_list = best_predictions.tolist()
                else:
                    # If 2D array, take first row
                    pred_list = best_predictions[0].tolist() if best_predictions.shape[0] > 0 else []
            elif isinstance(best_predictions, list):
                pred_list = best_predictions
            else:
                pred_list = []
            
            for i, obj_name in enumerate(objective_names):
                if i < len(pred_list):
                    pred_value = pred_list[i]
                    # Ensure it's a scalar value
                    if hasattr(pred_value, '__iter__') and not isinstance(pred_value, (int, float, str)):
                        pred_value = pred_value[0] if len(pred_value) > 0 else 0.0
                    st.write(f"- **{obj_name}**: `{float(pred_value):.4f}`")
                else:
                    st.write(f"- **{obj_name}**: `N/A`")
        
        # Show which models were used
        if hasattr(evaluator, 'get_best_model_per_objective'):
            best_models = evaluator.get_best_model_per_objective()
            if best_models:
                st.write("**Best Models Used:**")
                for obj_name, model_name in best_models.items():
                    st.write(f"- **{obj_name}**: {model_name}")
            else:
                st.write("No model metrics available (e.g., SDL measurements only).")

    def _safe_get_prediction(self, candidate: Dict, objective_name: str, objective_names: List[str]) -> float:
     """Safely extract prediction value for a specific objective - FIXED VERSION"""
     try:
        predictions = candidate['predictions']
        
        # Handle different prediction formats consistently
        if isinstance(predictions, list):
            if objective_name in objective_names:
                obj_idx = objective_names.index(objective_name)
                if obj_idx < len(predictions):
                    value = predictions[obj_idx]
                    return float(value) if not isinstance(value, (list, np.ndarray)) else float(value[0])
            # Fallback: return first prediction
            return float(predictions[0]) if predictions else 0.0
            
        elif isinstance(predictions, np.ndarray):
            if predictions.size == 0:
                return 0.0
                
            if len(predictions.shape) == 1:
                if objective_name in objective_names:
                    obj_idx = objective_names.index(objective_name)
                    if obj_idx < len(predictions):
                        return float(predictions[obj_idx])
                return float(predictions[0]) if len(predictions) > 0 else 0.0
            else:
                # 2D array - find the right objective
                if objective_name in objective_names:
                    obj_idx = objective_names.index(objective_name)
                    if obj_idx < predictions.shape[1]:
                        return float(predictions[0, obj_idx])
                return float(predictions[0, 0])  # Fallback to first prediction
                
        else:
            return 0.0
            
     except (IndexError, ValueError, TypeError) as e:
        print(f"Warning: Could not extract prediction for {objective_name}: {e}")
        return 0.0

    def _candidate_dataframe(self, result: OptimizationResult,
                             parameter_configs: List[Any],
                             objective_configs: List[Any]) -> pd.DataFrame:
        """Return a tidy dataframe with parameters, objectives and distance for plotting."""
        records = []
        objective_names = [o.name for o in objective_configs]
        parameter_names = [p.name for p in parameter_configs]

        for idx, cand in enumerate(result.all_candidates):
            rec = {
                'index': idx + 1,
                'distance': cand.get('distance', np.nan),
                'is_experimental': cand.get('is_experimental', False)
            }
            params = cand.get('parameters', {})
            for name in parameter_names:
                rec[name] = params.get(name, np.nan)

            obj_vals = cand.get('objective_values', {}) or {}
            for obj_name in objective_names:
                val = obj_vals.get(obj_name)
                if val is None:
                    val = self._safe_get_prediction(cand, obj_name, objective_names)
                rec[obj_name] = val
            records.append(rec)

        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    def _compute_pareto_fallback(self, df: pd.DataFrame,
                                 parameter_configs: List[Any],
                                 objective_configs: List[Any]) -> List[Dict[str, Any]]:
        """Compute Pareto front from available candidate dataframe when backend did not return one."""
        if df.empty or len(objective_configs) < 2:
            return []

        # Filter rows with all objectives present
        obj_names = [o.name for o in objective_configs]
        valid_df = df.dropna(subset=obj_names, how='any')
        pareto_points = []
        if valid_df.empty:
            return []

        def dominates(a, b):
            better_or_equal = True
            strictly_better = False
            for obj in objective_configs:
                a_val = a[obj.name]
                b_val = b[obj.name]
                direction = obj.type.value if hasattr(obj.type, 'value') else obj.type
                if direction == "minimize":
                    if a_val > b_val:
                        better_or_equal = False
                    if a_val < b_val:
                        strictly_better = True
                else:  # maximize / target -> treat as maximize on closeness
                    if a_val < b_val:
                        better_or_equal = False
                    if a_val > b_val:
                        strictly_better = True
            return better_or_equal and strictly_better

        rows = valid_df.to_dict(orient='records')
        for i, a in enumerate(rows):
            dominated = False
            for j, b in enumerate(rows):
                if i == j:
                    continue
                if dominates(b, a):
                    dominated = True
                    break
            if not dominated:
                pareto_points.append({
                    'parameters': {p.name: a.get(p.name, np.nan) for p in parameter_configs},
                    'outcomes': {obj.name: a.get(obj.name, np.nan) for obj in objective_configs}
                })
        return pareto_points


    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig
