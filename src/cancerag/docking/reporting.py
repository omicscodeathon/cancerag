import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_affinity_distribution(all_parsed_results: dict, output_dir: str):
    """Plots the distribution of binding affinities for each receptor."""
    plt.figure(figsize=(12, 7))
    for receptor, results in all_parsed_results.items():
        affinities = [r["best_affinity"] for r in results if "best_affinity" in r]
        if affinities:
            sns.kdeplot(affinities, label=receptor, fill=True, alpha=0.2)

    plt.xlabel("Binding Affinity (kcal/mol)")
    plt.ylabel("Density")
    plt.title("Distribution of Best Binding Affinities Across Receptors")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    output_path = os.path.join(output_dir, "affinity_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  - Saved affinity distribution plot to {output_path}")


def visualize_receptor_comparison(affinity_df: pd.DataFrame, output_dir: str):
    """Creates scatter plots comparing binding affinities between receptor pairs."""
    receptor_names = [col for col in affinity_df.columns if not col.startswith("bias_")]

    if len(receptor_names) < 2:
        return

    for i, r1 in enumerate(receptor_names):
        for r2 in receptor_names[i + 1 :]:
            plt.figure(figsize=(10, 10))

            # Drop rows where either receptor has no score
            plot_df = affinity_df[[r1, r2]].dropna()

            sns.scatterplot(data=plot_df, x=r1, y=r2, alpha=0.7)

            min_val = min(plot_df[r1].min(), plot_df[r2].min()) - 1
            max_val = max(plot_df[r1].max(), plot_df[r2].max()) + 1
            plt.plot(
                [min_val, max_val],
                [min_val, max_val],
                "r--",
                alpha=0.8,
                label="Equal Affinity",
            )

            plt.xlabel(f"{r1} Binding Affinity (kcal/mol)")
            plt.ylabel(f"{r2} Binding Affinity (kcal/mol)")
            plt.title(f"Binding Affinity Comparison: {r1} vs {r2}")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.axis("square")

            output_path = os.path.join(output_dir, f"comparison_{r1}_vs_{r2}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  - Saved comparison plot for {r1} vs {r2} to {output_path}")


def generate_html_report(
    all_parsed_results: dict, affinity_df: pd.DataFrame, output_dir: str
):
    """Generates a comprehensive HTML report of the docking results."""
    report_file = os.path.join(output_dir, "docking_report.html")

    with open(report_file, "w") as f:
        # --- HTML Header ---
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>GPCR Docking Results Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 40px; line-height: 1.6; background-color: #f8f9fa; color: #212529; }}
        h1, h2, h3 {{ color: #0056b3; border-bottom: 2px solid #dee2e6; padding-bottom: 10px;}}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
        th {{ background-color: #e9ecef; }}
        tr:nth-child(even) {{ background-color: #fdfdfe; }}
        .container {{ max-width: 1200px; margin: auto; }}
        .figure {{ margin: 30px 0; text-align: center; background-color: #ffffff; padding: 20px; border-radius: 5px; box-shadow: 0 2px 3px rgba(0,0,0,0.1);}}
        .figure img {{ max-width: 80%; border: 1px solid #dee2e6; }}
        .caption {{ font-style: italic; margin-top: 10px; color: #6c757d;}}
    </style>
</head>
<body><div class="container">
    <h1>GPCR Docking Analysis Report</h1>
    <p>Generated on {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
""")

        # --- Top Compounds Section ---
        f.write("<h2>Top Compounds by Receptor</h2>")
        for receptor, results in all_parsed_results.items():
            f.write(f"<h3>{receptor} (Top 10)</h3>")
            if results:
                top_10_df = pd.DataFrame(results).head(10)
                f.write(top_10_df.to_html(index=False, classes="table table-striped"))
            else:
                f.write("<p>No successful docking results for this receptor.</p>")

        # --- Visualizations Section ---
        f.write("<h2>Visual Analysis</h2>")
        f.write(
            '<div class="figure"><img src="affinity_distribution.png" alt="Binding affinity distributions"><div class="caption">Figure 1: Distribution of best binding affinities across different receptors.</div></div>'
        )

        receptor_names = [
            col for col in affinity_df.columns if not col.startswith("bias_")
        ]
        for i, r1 in enumerate(receptor_names):
            for r2 in receptor_names[i + 1 :]:
                img_path = f"comparison_{r1}_vs_{r2}.png"
                f.write(
                    f'<div class="figure"><img src="{img_path}" alt="Comparison of {r1} vs {r2}"><div class="caption">Figure: Binding affinity comparison between {r1} and {r2}.</div></div>'
                )

        # --- Biased Ligands Section ---
        f.write("<h2>Biased Ligands Analysis (Top 10 per category)</h2>")
        bias_cols = [col for col in affinity_df.columns if col.startswith("bias_")]
        for bias_col in bias_cols:
            r1, r2 = bias_col.replace("bias_", "").split("_vs_")

            # Biased towards r1 (more negative score)
            f.write(f"<h3>Biased towards {r1} (vs {r2})</h3>")
            biased_r1_df = affinity_df.sort_values(by=bias_col, ascending=True).head(10)
            f.write(
                biased_r1_df[[r1, r2, bias_col]].to_html(classes="table table-striped")
            )

            # Biased towards r2 (more positive score)
            f.write(f"<h3>Biased towards {r2} (vs {r1})</h3>")
            biased_r2_df = affinity_df.sort_values(by=bias_col, ascending=False).head(
                10
            )
            f.write(
                biased_r2_df[[r1, r2, bias_col]].to_html(classes="table table-striped")
            )

        # --- HTML Footer ---
        f.write("</div></body></html>")

    print(f"  - Successfully generated HTML report: {report_file}")
