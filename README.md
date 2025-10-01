# Mini_Project: A Large Language Model Dataset Analysis from the HuggingFace Leaderboard 2024-2025

## Introduction

Large Language Model, like ChatGPT, Claude, Deepseek, and many others have been popular and an increasing trend since 2022, they will continue to be a part of this generation's everyday's life and it is crucial to know how they work and the keywords associated with them. Their core keywords like the parameters, benchmark scores, models fine-tuning, and techniques like Mixture of Experts (MoE), which is used for efficiency and how its important will be explored on as well.

This project aims to explore the HuggingFace leaderboard dataset, targeting the parameters, the average benchmark scores of MoE models vs non-MoE models, and the CO2 costs of the models and visualizing them to find insights.

## Table of Contents

[Key Concepts and Terminology](#key-concepts-and-terminology)

[Data Preparation](#data-preparation)

[Workflows and Insights](#workflows-and-insights)

[Conclusion](conclusion)

## Key Concepts and Terminology

This section will explain the keywords that will be used in this project.

**Parameters**: Parameters refer to the numerical values within a large language model that are learned during training. They determine how the model processes and generates text by adjusting the weights and biases of its neural network. In general, a higher number of parameters increases a model’s capacity to capture complex patterns in data, although this does not always guarantee improved performance or efficiency.

**Benchmark Scores**: Benchmark scores represent standardized performance evaluations of large language models across a variety of tasks, such as reasoning, language understanding, and problem-solving. They provide a common ground for comparing different models by aggregating results from widely recognized datasets and tests. Average benchmark scores are often used to summarize overall capability, though specific benchmarks may highlight strengths and weaknesses in particular domains.

**Mixture of Experts (MoE)**: The Mixture of Experts (MoE) is an architectural approach designed to improve efficiency in large models. Instead of activating all parts of the model simultaneously, MoE structures contain multiple specialized “experts,” with only a subset being used for each input. This allows models to scale to very large sizes while maintaining lower computational costs during inference, balancing performance and efficiency.

**Fine-tuning**: Fine-tuning is the process of adapting a pre-trained language model to specific tasks or domains by continuing its training on targeted datasets. This technique allows models to specialize beyond their general training, improving relevance and accuracy in specific tasks.

## Data Preparation

The dataset is from the HuggingFace Leaderboard Dataset

Data and Library imports:

```
from datasets import load_dataset
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import altair as alt

dataset = load_dataset("open-llm-leaderboard/contents", split="train")
```

Set the DataFrame through Pandas, drop any unnecessary columns, set date format, and rename any complicating columns name.

```
# Convert to pandas
df = pd.DataFrame(dataset)

# Set display column to get the overview and set the submission date to date format
pd.set_option('display.max_columns', None)
df['Submission Date'] = pd.to_datetime(df['Submission Date'])

#Remove trivial columns
df = df.drop(columns=['T', 'Weight type', 'Model sha', 'Hub License', 'Chat Template', 'Generation', 'Precision', 'Model', 'Available on the hub', 'Merged'])

#Remove individual scores as specializations of models are not the main focus of the project
df = df.drop(columns=['IFEval Raw','IFEval','BBH Raw', 'MATH Lvl 5 Raw', 'MATH Lvl 5', 'GPQA Raw', 'GPQA', 'MUSR Raw', 'MUSR', 'BBH', 'MMLU-PRO Raw', 'MMLU-PRO'])
df

#Rename for easier inputs
df = df.rename(columns={"Average ⬆️": "Average"})
df = df.rename(columns={"CO₂ cost (kg)": "CO2 Cost (kg)"})
df = df.rename(columns={"Hub ❤️": "Ratings"})

#Normalize Data

str_cols = ['fullname', 'eval_name', 'Type', 'Architecture', 'Base Model']
for col in str_cols:
    df[col] = df[col].str.strip().str.lower()

df['#Params (B)'] = df['#Params (B)'].round(3)
df['CO2 Cost (kg)'] = df['CO2 Cost (kg)'].round(2)
```

Remove Duplicates and Check for any errors such as models having 0.0 as their parameters.

```
df.drop_duplicates(subset=['fullname', '#Params (B)', 'Type'], keep='first', inplace=True)

#Check models where parameters = 0
df.loc[(df['#Params (B)'] == 0)]

#For Parameters, add parameters number according model name.
df.loc[341, '#Params (B)'] = 8.0
df.loc[944, '#Params (B)'] = 14.0
df.loc[2123, '#Params (B)'] = 7.0
df.loc[2936, '#Params (B)'] = 7.0

#Drop MoE model without parameters
df = df[df['#Params (B)'] > 0.0]

#Reset index
df = df.reset_index(drop=True)
```

## Workflows and Insights

In this section, the author's workflow and analysis will be explored. Upon initial inspection, there are many useful and interesting columns such as: <ins>Parameters</ins>, <ins>Average</ins> (Benchmark Scores), <ins>Ratings</ins> (from the HuggingFace Hub), <ins>Submission Date</ins>, <ins>Official Provider</ins>, and the <ins>CO2 Cost</ins>.

For easier analysis, the author have categorized parameters into 3 main categories as follows: Small (0-9 Billion Parameters Models), Medium (9-40 Billion Parameters Models), and Large (>40 Billion Parameters Models)

```
bins = [0, 9, 40, float('inf')]
labels = ['Small', 'Medium', 'Large']

df['Model Size'] = pd.cut(df['#Params (B)'], bins=bins, labels=labels, right=True, include_lowest=True)
```

Then, the author explore through visualizations the data to find trends, starting from the basic Average Performance (Benchmark Score) per Parameters

<img width="438" height="487" alt="image" src="https://github.com/user-attachments/assets/6f5494a4-3909-49c9-ac0d-ca1257faea7c" />


 Figure 1. Average Performance (Benchmark Score) per Parameters

In Figure 1, to reduce smaller models distorting the ratio, models that is smaller than <1 billion Parameters have been cut off to make small model section more accurate.

This shows the diminishing return of the more parameters, the less benchmark score will be gained per parameters as more parameters are added to the model, which is as expected.

<img width="543" height="454" alt="image" src="https://github.com/user-attachments/assets/f4e499c1-b30c-4594-9f5c-aaac5cca69a7" />

Figure 2.

Figure 2 help elaborates more clearly the trend of how benchmark scores drop when adding more parameters. The X axis uses log scales as to not clump everything to the left and the Y axis shows the efficiency of score per parameter.

<br>
<br>
<br>

Now, in the HuggingFace dataset, there is various fine-tuned models (Official Provider = "False"), and most of the fine-tuning results in better performance scores and better CO2 cost efficiency as follows.

<img width="580" height="608" alt="image" src="https://github.com/user-attachments/assets/9d480170-db00-4a59-b5a1-cba004c35a74" />

<img width="573" height="452" alt="image" src="https://github.com/user-attachments/assets/baf64c53-9bfb-4897-b314-c9d1af1cf58c" />

 Figure 3. CO2 Cost Performance Scatterplot with Trendline

Figure 3 shows how the fine-tuned models from the community outperforms the Official Released models but is it because the community prefer to use only smaller, more efficient models?
The author also took the ratio of each group in proportion to the "large" group model.

<img width="649" height="541" alt="image" src="https://github.com/user-attachments/assets/da43d80b-2bd2-4afa-8f83-7bf640b72d4f" />

 Figure 4. Unofficial Models Count Ratio

<img width="649" height="541" alt="image" src="https://github.com/user-attachments/assets/afa1a26d-1e66-4095-9d76-b95f811686fb" />

 Figure 5. Official Models Count Ratio

Figure 4 and 5 shows there is a unequality in the ratio of data between Official and Unofficial models, and the unofficial models are skewed toward the smaller, more efficient models. To make it more accurate, normalize by Parameters to make the comparison more accurate as will be seen in Figure 6.

<img width="497" height="486" alt="image" src="https://github.com/user-attachments/assets/a606c400-f15e-40f3-8b19-ccd3a89acda3" />

 Figure 6. Performance of Unofficial and Official Models by Model Size

<img width="950" height="307" alt="image" src="https://github.com/user-attachments/assets/37e8c66f-7958-4eba-8e13-250324bf5e4e" />


 Figure 7. CO2 per Parameters by Model Size

Interestingly, medium-sized models demonstrate better CO2 efficiency per parameter than even small models. This suggests that there may be an optimal size range where models balance training overhead and parameter scaling more effectively, before large-scale inefficiencies dominate.


<img width="1067" height="643" alt="image" src="https://github.com/user-attachments/assets/3ca249ba-8d10-41bc-8416-80366dc70b3c" />

 Figure 8. Qwen Model Performance (Unofficial vs Official)

For further accuracy, narrowing down to only one main provider (Qwen), we can still see the same trend where of unofficial models outperforming the official, which can then be concluded that fine-tuned models will provide more efficiency in terms of CO2 cost, Parameters, and Benchmark Scores.

<img width="1292" height="369" alt="image" src="https://github.com/user-attachments/assets/469b05cb-023a-41f6-a975-6b65f325e4c7" />

 Figure 9.

After inspecting the Official Providers Models and comparing it to community fine-tuned models, the dataset also have Mixture of Experts column which we have not used.
Unfortunately, some of the Parameters for MoE models contains the overall parameters, instead of the active parameters when in-used, which derives the numerical advantage of the MoE models as shown in Figure 9.


<img width="1130" height="640" alt="image" src="https://github.com/user-attachments/assets/03fad89a-baa0-47fe-a086-d74c22c56880" />

 Figure 10. MoE vs Dense CO2 Efficiency 


From Figure 10, in terms of CO2 Efficiency, the MoE models are on par with fine-tuned models


## Conclusion

In conclusion.

1. The increase in benchmark scores diminishes as the number of parameters went up.

2. The fine-tuned models consistently outperforms pretrained official models in terms of CO2 efficiency, benchmark scores and parameters efficiency.
In the future, if rules and regulations regarding Large Language Model power consumption became a problem, fine-tuning can reduce and optimize CO2 cost while maintaining or even increasing capabilities.

3. MoE parameters in the datasets are inconsistent between active parameters and overall parameters, and CO2 efficiency is comparable to fine-tuned models. A conclusion can not made that fine-tuned MoE models are most efficient.
Getting a more accurate MoE model parameters would be the next step for a more accurate conclusion.


## Additional Info
Hugging Face Leaderboard: Link https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/
CO2 Cost Calculation (HuggingFace): Link https://huggingface.co/docs/leaderboards/open_llm_leaderboard/emissions








