import json
import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_and_plot_evals(json_path="eval_regression_history.json"):
    if not os.path.exists(json_path):
        print(f"[!] Error: '{json_path}' not found.")
        return

    # 1. Load JSON history
    with open(json_path, "r") as f:
        history = json.load(f)

    if not history:
        print("[!] Evaluation history is empty.")
        return

    # 2. Convert to Pandas DataFrame
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # 3. Print Statistical Summary Report
    print("=" * 65)
    print("         RAGAS EVALUATION REGRESSION REPORT")
    print("=" * 65)
    print(f"Total Evaluation Runs : {len(df)}")
    print(f"Tracking Window       : {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("-" * 65)
    
    metrics = ["mean_faithfulness", "mean_answer_relevancy", "mean_context_precision"]
    for m in metrics:
        if m in df.columns:
            metric_name = m.replace("mean_", "").replace("_", " ").title()
            print(f"Metric: {metric_name}")
            print(f"  - Latest Score  : {df[m].iloc[-1]:.4f}")
            print(f"  - Peak Score    : {df[m].max():.4f}")
            print(f"  - Average Score : {df[m].mean():.4f}")
            print("-" * 65)

    # 4. Generate Trend Charts
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

    plt.plot(df["timestamp"], df["mean_faithfulness"], marker='o', label="Faithfulness", color="#3b82f6", linewidth=2.5)
    plt.plot(df["timestamp"], df["mean_answer_relevancy"], marker='s', label="Answer Relevancy", color="#10b981", linewidth=2.5)
    plt.plot(df["timestamp"], df["mean_context_precision"], marker='^', label="Context Precision", color="#f59e0b", linewidth=2.5)

    plt.title("GraphRAG Agent Performance Regression Over Time", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Execution Timestamp", fontsize=12)
    plt.ylabel("Metric Score (0.0 - 1.0)", fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.xticks(rotation=30)
    plt.legend(loc="lower left", fontsize=10, frameon=True)
    plt.tight_layout()

    # Save chart image
    chart_output = "evaluation_trend_report.png"
    plt.savefig(chart_output, dpi=300)
    print(f"\n[✓] Trend chart successfully saved as '{chart_output}'")
    plt.show()

if __name__ == "__main__":
    analyze_and_plot_evals()
