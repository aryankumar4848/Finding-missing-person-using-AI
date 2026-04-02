import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ResultsVisualizer:
    """
    Generates academic-style graphs validating the performance margins
    between the 3D Mesh pipeline and the ArcFace Baseline over LFW degraded metrics.
    """
    def __init__(self, results_csv: str = "experiments/multilevel_results.csv", output_dir: str = "experiments/plots"):
        self.results_csv = results_csv
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load constraints
        if not os.path.exists(results_csv):
            raise FileNotFoundError(f"Missing {results_csv}. Please run the experiment suite first.")
            
        self.df = pd.read_csv(results_csv)
        
        # Aesthetics
        sns.set_theme(style="whitegrid")
        self.colors = {"Mesh": "#1f77b4", "ArcFace": "#d62728"} # Blue for Mesh, Red for Baseline

    def get_degradation_data(self, degradation: str):
        """Filters DataFrame isolating a specific evaluation parameter."""
        data = self.df[self.df['degradation_type'] == degradation].copy()
        # Convert severity levels to numeric for proper structural x-axis scaling
        data['severity_level'] = pd.to_numeric(data['severity_level'])
        return data.sort_values(by="severity_level")

    def _plot_metric_vs_degradation(self, metric: str, y_label: str, title_prefix: str, output_prefix: str):
        """Generic generator iterating over Blur, Noise, Low Light, Occlusion."""
        degradations = ['blur', 'noise', 'low_light', 'occlusion']
        
        for deg in degradations:
            data = self.get_degradation_data(deg)
            if data.empty:
                continue
                
            plt.figure(figsize=(8, 6))
            sns.lineplot(
                data=data, 
                x='severity_level', 
                y=metric, 
                hue='method',
                marker='o',
                palette=self.colors,
                linewidth=2.5,
                markersize=8
            )
            
            plt.title(f"{title_prefix} - {deg.replace('_', ' ').title()}", fontsize=14, fontweight='bold')
            plt.xlabel(f"{deg.title()} Severity Level", fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.legend(title="Method", title_fontsize='13', fontsize='11')
            plt.ylim(0, 1.05)
            
            # Additional structural lines highlighting critical FPR boundaries conditionally
            if metric == "fpr":
                plt.axhline(y=0.1, color='black', linestyle='--', alpha=0.5, label='FPR Hazard Line')
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{output_prefix}_{deg}.png"), dpi=300)
            plt.close()

    def plot_precision_curves(self):
        print("Generating Precision Curves...")
        self._plot_metric_vs_degradation('precision', 'Precision (Accuracy of Matches)', 'Precision vs Severity', 'precision')

    def plot_fpr_curves(self):
        print("Generating False Positive Rate Curves...")
        self._plot_metric_vs_degradation('fpr', 'False Positive Rate', 'False Positive Danger Bounds', 'fpr')

    def plot_accuracy_bar_chart(self):
        """
        Maps Clean dataset performance vs Aggregated statistical performance of all anomalies.
        """
        print("Generating Global Accuracy Bar Chart...")
        
        # 1. Clean Data
        clean_df = self.df[self.df['degradation_type'] == 'clean'].copy()
        clean_df['Dataset'] = 'Clean (LFW Original)'
        
        # 2. Degraded Aggregation Average
        degraded_df = self.df[self.df['degradation_type'] != 'clean'].copy()
        
        # We group by Method mathematically 
        agg_degraded = degraded_df.groupby('method')['accuracy'].mean().reset_index()
        agg_degraded['Dataset'] = 'Degraded (Synthetic CCTV Average)'
        
        # Concat constraints
        plot_df = pd.concat([clean_df[['method', 'accuracy', 'Dataset']], agg_degraded], ignore_index=True)
        
        plt.figure(figsize=(9, 6))
        ax = sns.barplot(
            data=plot_df, 
            x='Dataset', 
            y='accuracy', 
            hue='method',
            palette=self.colors,
            edgecolor='black'
        )
        
        plt.title('Global System Accuracy: Clean Optics vs Surveillance Bounds', fontsize=15, fontweight='bold')
        plt.xlabel('', fontsize=12)
        plt.ylabel('Evaluation Accuracy', fontsize=12)
        plt.ylim(0, max(plot_df['accuracy'].max() * 1.2, 0.5)) 
        plt.legend(title="Method")
        
        # Annotate bars algorithmically
        for i in ax.containers:
            ax.bar_label(i, fmt='%.3f', padding=3, fontsize=10)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "accuracy_comparison_bar.png"), dpi=300)
        plt.close()

    def generate_all(self):
        self.plot_precision_curves()
        self.plot_fpr_curves()
        self.plot_accuracy_bar_chart()
        print(f"All formal evaluation graphs successfully exported to {self.output_dir}/")

if __name__ == "__main__":
    visualizer = ResultsVisualizer()
    visualizer.generate_all()
