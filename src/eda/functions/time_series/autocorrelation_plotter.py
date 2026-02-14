import matplotlib.pyplot as plt


class AutocorrelationPlotter:
    """
    Plots autocorrelation function (ACF).
    Single responsibility: visualize temporal dependence.
    Public API: run().
    """
    
    def run(
        self,
        acf_dict: dict,
        title: str = "Autocorrelation (ACF)",
        figsize: tuple[int, int] = (10, 5)
    ) -> plt.Figure:
        """
        Public API. Creates ACF plot with confidence bounds.
        
        Args:
            acf_dict: dict from AutocorrelationAnalyzer.run()
            title: plot title
            figsize: figure size
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        lags = acf_dict['lags']
        acf_values = acf_dict['acf_values']
        confidence_bound = acf_dict['confidence_bound']
        
        # Plot ACF as vertical bars
        ax.bar(lags, acf_values, width=0.3, color='steelblue', alpha=0.7)
        
        # Add confidence bounds
        ax.axhline(y=confidence_bound, color='r', linestyle='--', 
                  linewidth=1.5, label=f'95% confidence (±{confidence_bound:.3f})')
        ax.axhline(y=-confidence_bound, color='r', linestyle='--', linewidth=1.5)
        ax.axhline(y=0, color='black', linewidth=0.8)
        
        # Highlight significant lags
        sig_lags = acf_dict['significant_lags']
        if sig_lags:
            sig_text = f"Significant lags: {sig_lags[:10]}"  # show first 10
            if len(sig_lags) > 10:
                sig_text += f"... (+{len(sig_lags)-10} more)"
            ax.text(0.98, 0.98, sig_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        
        ax.set_xlabel('Lag (weeks)', fontsize=11)
        ax.set_ylabel('Autocorrelation', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xlim(-1, len(lags))
        
        fig.tight_layout()
        return fig