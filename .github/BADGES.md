# GitHub Workflow Status Badges

Add these badges to your README.md to show the status of your workflows:

## Main Workflows

```markdown
[![CI/CD Pipeline](https://github.com/yourusername/disease-outbreak-prediction/workflows/🦠%20Disease%20Outbreak%20Prediction%20CI%2FCD/badge.svg)](https://github.com/yourusername/disease-outbreak-prediction/actions/workflows/ci.yml)

[![Model Monitoring](https://github.com/yourusername/disease-outbreak-prediction/workflows/📊%20Model%20Performance%20Monitoring/badge.svg)](https://github.com/yourusername/disease-outbreak-prediction/actions/workflows/model-monitoring.yml)

[![Release](https://github.com/yourusername/disease-outbreak-prediction/workflows/🚀%20Release/badge.svg)](https://github.com/yourusername/disease-outbreak-prediction/actions/workflows/release.yml)

[![Documentation](https://github.com/yourusername/disease-outbreak-prediction/workflows/📚%20Documentation/badge.svg)](https://github.com/yourusername/disease-outbreak-prediction/actions/workflows/documentation.yml)

[![Dependency Updates](https://github.com/yourusername/disease-outbreak-prediction/workflows/🔄%20Dependency%20Updates/badge.svg)](https://github.com/yourusername/disease-outbreak-prediction/actions/workflows/dependency-updates.yml)
```

## All-in-One Status Row

```markdown
![CI](https://github.com/yourusername/disease-outbreak-prediction/workflows/🦠%20Disease%20Outbreak%20Prediction%20CI%2FCD/badge.svg) ![Models](https://github.com/yourusername/disease-outbreak-prediction/workflows/📊%20Model%20Performance%20Monitoring/badge.svg) ![Docs](https://github.com/yourusername/disease-outbreak-prediction/workflows/📚%20Documentation/badge.svg) ![Release](https://github.com/yourusername/disease-outbreak-prediction/workflows/🚀%20Release/badge.svg)
```

## Individual Badges for Specific Sections

### For Installation Section
```markdown
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Tests](https://github.com/yourusername/disease-outbreak-prediction/workflows/🦠%20Disease%20Outbreak%20Prediction%20CI%2FCD/badge.svg)](https://github.com/yourusername/disease-outbreak-prediction/actions/workflows/ci.yml)
```

### For Models Section
```markdown
[![Model Performance](https://github.com/yourusername/disease-outbreak-prediction/workflows/📊%20Model%20Performance%20Monitoring/badge.svg)](https://github.com/yourusername/disease-outbreak-prediction/actions/workflows/model-monitoring.yml)
```

### For Documentation Section
```markdown
[![Documentation](https://github.com/yourusername/disease-outbreak-prediction/workflows/📚%20Documentation/badge.svg)](https://github.com/yourusername/disease-outbreak-prediction/actions/workflows/documentation.yml)
[![Docs](https://img.shields.io/badge/docs-available-brightgreen?style=flat-square)](https://yourusername.github.io/disease-outbreak-prediction/)
```

## Usage Instructions

1. Replace `yourusername` with your actual GitHub username
2. Replace `disease-outbreak-prediction` with your repository name if different
3. Add the badges to appropriate sections of your README.md
4. The badges will automatically update to show current workflow status

## Custom Badge Colors

You can customize badge colors by adding parameters:
- `?style=flat-square` - Different style
- `?color=red` - Red color for failures
- `?color=green` - Green color for success
- `?color=yellow` - Yellow for warnings

## Example in Context

```markdown
## 🚀 Quick Start

[![CI Status](https://github.com/yourusername/disease-outbreak-prediction/workflows/🦠%20Disease%20Outbreak%20Prediction%20CI%2FCD/badge.svg)](https://github.com/yourusername/disease-outbreak-prediction/actions/workflows/ci.yml)

Follow these steps to get started...
```