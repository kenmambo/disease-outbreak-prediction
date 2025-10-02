import json
from datetime import datetime

def generate_report():
    """
    Generates a markdown performance report from model_performance.json.
    """
    try:
        with open('model_performance.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: model_performance.json not found. Run evaluation first.")
        return

    report = f"""# 📊 Model Performance Report - {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

## 🤖 Model Accuracies

| Model | Accuracy | MAE | RMSE | Status |
|---|---|---|---|---|
| **LSTM** | {results.get('lstm_accuracy', 0):.2f}% | {results.get('lstm_mae', 0):.4f} | {results.get('lstm_rmse', 0):.4f} | {'✅ Good' if results.get('lstm_accuracy', 0) >= 85 else '⚠️ Needs Attention'} |
| **Random Forest** | {results.get('rf_accuracy', 0):.2f}% | {results.get('rf_mae', 0):.4f} | {results.get('rf_rmse', 0):.4f} | {'✅ Good' if results.get('rf_accuracy', 0) >= 80 else '⚠️ Needs Attention'} |

## 📈 Feature Importance (Random Forest)
"""

    if results.get('feature_importance'):
        for feat in results['feature_importance'][:5]:  # Top 5 features
            report += f"- **{feat.get('feature', 'N/A')}**: {feat.get('importance', 0):.4f}\\n"
    else:
        report += "- No feature importance data available\\n"

    report += f"""
## 🎯 Performance Thresholds
- **LSTM Threshold**: 85.0% (Current: {results.get('lstm_accuracy', 0):.2f}%)
- **Random Forest Threshold**: 80.0% (Current: {results.get('rf_accuracy', 0):.2f}%)

## 🔄 Recommendation
{'🚨 **Model retraining recommended** - Performance below threshold' if results.get('needs_retraining') else '✅ **Models performing well** - No action needed'}

---
*Generated automatically by Model Performance Monitoring workflow*
"""

    with open('performance_report.md', 'w') as f:
        f.write(report)

    print("✅ Performance report generated successfully.")
    print(report)

if __name__ == "__main__":
    generate_report()