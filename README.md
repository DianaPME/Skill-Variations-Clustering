# Skill Variations Clustering

This research project implements a model designed to normalize and group semantically and lexically similar terms, generating predicted clusters and associated topic keywords for each group. It is important to consider that this implementation uses a domain-adapted transformer-based language model trained with job postings from the Mexican automotive industry.

**SkillVariationClustering Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | The file with the text to be grouped. |
| `text_column` | `str` | The name of the column with the text you want to group |
| `use_vgcn` | `bool` | Flag to use fused embeddings with VGCN, False to use only embeddings |
| 'projection_matrix_path' | `str` | Path to the pkl file of the projection matrix, None if use_vgcn=False |

## SkillVariationClustering Running example 
```bash
file_path = '/content/automative_data.xlsx'
df= pd.read_excel(file_path)
text_column = 'Variations'
use_vgcn= True
projection_matrix_path= '/content/projection_matrix.pkl'
clustering_model = SkillVariationClustering(df, text_column, use_vgcn, projection_matrix_path)
predicted_labels, topics= clustering_model.cluster_texts()
```
The predicted labels are stored in a NumPy array, where each element represents the cluster ID assigned to each text.
Topics are a list of words referring to each ID of each text.

Running the model adds two columns to the df: one named 'processed_text', with the clean text, other named'normalized' with the normalized text, as well as 'cluster_id' with the id of each text, and 'topics' with the topic words associated with each text.

**TextCleaning Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `df` | `pd.DataFrame` | The file with the text to be cleaned. |
| `text_column` | `str` | The name of the column with the text you want to clean |

## SkillVariationClustering Running example 
```bash
file_path = '/content/automative_data.xlsx'
df= pd.read_excel(file_path)
text_column = 'Variations'
df_clean= TextCleaning(df, text_column)
```
