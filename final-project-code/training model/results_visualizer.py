import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

class ResultsVisualizer:
    """Handles plotting of evaluation results stored in a dictionary."""

    def __init__(self, results_data: Dict):
        """
        Initializes the ResultsVisualizer.

        Args:
            results_data: Dictionary containing evaluation results (model_name -> scores).
                          Example: {'model_A': {'validation_score': 85.0, 'test_score': 82.5}, ...}
        """
        self.results_data = results_data
        self.df_results = self._prepare_dataframe()
        print("ResultsVisualizer initialized.")
        if self.df_results is not None:
            print("Results DataFrame prepared:")
            print(self.df_results)
        else:
            print("Warning: Could not prepare DataFrame from results data.")

    def _prepare_dataframe(self) -> Optional[pd.DataFrame]:
        """Converts the results dictionary into a pandas DataFrame."""
        if not self.results_data:
            return None
        try:
            df = pd.DataFrame.from_dict(self.results_data, orient='index')
            # Filter out models with error scores (-1.0) before plotting
            df = df[(df['validation_score'] >= 0) & (df['test_score'] >= 0)]
            df = df.reset_index().rename(columns={'index': 'model'})
            return df
        except Exception as e:
            print(f"Error converting results to DataFrame: {e}")
            return None

    def plot_comparison_bar_chart(self, title: str = 'Model Performance Comparison', figsize: tuple = (12, 7), palette: str = 'viridis'):
        """Plots a bar chart comparing validation and test scores across models."""
        if self.df_results is None or self.df_results.empty:
            print("No valid data available for plotting.")
            return

        # Melt the DataFrame for easier plotting with seaborn
        try:
            df_melted = pd.melt(self.df_results, id_vars=['model'], var_name='split', value_name='score')
            df_melted['split'] = df_melted['split'].str.replace('_score', '') # Clean up split names
        except Exception as e:
            print(f"Error melting DataFrame: {e}")
            return

        # Set plot style
        sns.set_theme(style="whitegrid")

        # Plotting
        plt.figure(figsize=figsize)
        ax = sns.barplot(data=df_melted, x='model', y='score', hue='split', palette=palette)

        plt.title(title, fontsize=16)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(105, df_melted['score'].max() * 1.1)) # Adjust ylim dynamically
        plt.legend(title='Dataset Split')

        # Add score labels
        try:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)
        except Exception as e:
            print(f"Error adding bar labels: {e}")

        plt.tight_layout()
        plt.show()

    def plot_split_bar_chart(self, split_name: str, title: Optional[str] = None, figsize: tuple = (10, 6), palette: str = 'Blues_d'):
        """Plots a bar chart for a specific split (e.g., 'validation' or 'test')."""
        if self.df_results is None or self.df_results.empty:
            print("No valid data available for plotting.")
            return

        score_column = f"{split_name}_score"
        if score_column not in self.df_results.columns:
            print(f"Error: Score column '{score_column}' not found in DataFrame.")
            return

        # Set plot style
        sns.set_theme(style="whitegrid")

        # Plotting
        plt.figure(figsize=figsize)
        ax = sns.barplot(data=self.df_results, x='model', y=score_column, palette=palette)

        plot_title = title if title else f'{split_name.capitalize()} Accuracy Comparison'
        plt.title(plot_title, fontsize=15)
        plt.xlabel('Model', fontsize=11)
        plt.ylabel('Accuracy (%)', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(105, self.df_results[score_column].max() * 1.1)) # Adjust ylim dynamically

        # Add score labels
        try:
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f%%', padding=3)
        except Exception as e:
            print(f"Error adding bar labels: {e}")

        plt.tight_layout()
        plt.show()

print("ResultsVisualizer loaded.")
