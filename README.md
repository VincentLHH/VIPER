# VIPER: Variant Impact Predictor with Expert Reasoning

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Inborn Errors of Immunity (IEI) encompass over 500 rare genetic disorders characterized by significant phenotypic and genotypic heterogeneity, presenting substantial challenges for molecular diagnosis. Existing variant prioritization tools face limitations in interpreting complex genomic data alongside unstructured phenotypes, which restricts their diagnostic efficacy in clinical settings. VIPER (Variant Impact Predictor with Expert Reasoning) is an open-source Large Language Model (LLM) framework designed for pathogenicity prediction and phenotype-genotype association analysis in the context of rare diseases.

Through strategic fine-tuning on curated IEI variant datasets and optimized prompting engineering, VIPER has demonstrated remarkable performance metrics, achieving a 92.20% diagnostic yield and a 92.64% Top-10 recall rate in an independent cohort of 205 IEI patients. When applied to 45 undiagnosed cases, VIPER successfully identified candidate causative variants within recently reported IEI genes such as ALPK1 and NBEAL2, highlighting its capability to recognize emerging gene-disease associations beyond its training scope. Further analyses have shown that VIPER's predictions are driven by systematic integration of multi-dimensional variant annotations, with gene-disease association evidence playing a critical role in enhancing diagnostic accuracy.

To validate its generalizability, VIPER was also applied to Inherited Retinal Disease (IRD), resulting in a 98.92% Top-10 recall in an independent cohort of 61 patients. 

## Features

- **High Diagnostic Yield**: Achieves up to 92.20% diagnostic yield in IEI patients.
- **Top-10 Recall Performance**: Ensures a high recall rate of top-10 predictions.
- **Emerging Gene-Disease Associations**: Capable of identifying new gene-disease relationships not covered in training data.
- **Modular Architecture**: Designed for easy adaptation to other monogenic disorders.
- **Open Source**: Released under the Apache 2.0 license to encourage community contribution and further development.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For more information, please contact [liuhaohan@genomics.cn].

---

*VIPER represents a significant step forward in leveraging LLM-based approaches for accelerating molecular diagnosis across various rare disease contexts.*
