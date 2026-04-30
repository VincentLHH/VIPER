# VIPER: Variant Impact Predictor with Expert Reasoning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Inborn Errors of Immunity (IEI) encompass over 500 rare genetic disorders characterized by significant phenotypic and genotypic heterogeneity, presenting substantial challenges for molecular diagnosis. Existing variant prioritization tools face limitations in interpreting complex genomic data alongside unstructured phenotypes, which restricts their diagnostic efficacy in clinical settings. VIPER (Variant Impact Predictor with Expert Reasoning) is an open-source Large Language Model (LLM) framework designed for pathogenicity prediction and phenotype-genotype association analysis in the context of rare diseases.

## Features

- **High Diagnostic Yield**: Achieves up to 92.20% diagnostic yield in IEI patients.
- **Top-10 Recall Performance**: Ensures a high recall rate of top-10 predictions.
- **Emerging Gene-Disease Associations**: Capable of identifying new gene-disease relationships not covered in training data.
- **Modular Architecture**: Designed for easy adaptation to other monogenic disorders.

## Environment Setup

VIPER requires Python >= 3.10 and the `swift==3.6.0` package ([A framework for LLM fine-tuning and inference](https://github.com/modelscope/ms-swift)). We recommend using Conda to create a reproducible environment that works across different CPU architectures (x86_64, aarch64).

### Files

**`environment.yml`**
```yaml
name: viper
channels:
  - conda-forge
  - defaults
dependencies:
  - python>=3.10
  - pip
  - pip:
      - -r requirements.txt
```

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For more information, please contact liuhaohan@genomics.cn.

---

*VIPER represents a significant step forward in leveraging LLM-based approaches for accelerating molecular diagnosis across various rare disease contexts.*
