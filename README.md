This reposi contains code used for the JEDM paper.
To replicate the results:
1. Following the instruction in this page to download the data https://sites.google.com/ncsu.edu/csedm-dc-2021/home?authuser=0  
Put the data under the data folder  

2. Preprocess data  
2.1 Run organize_data.py   
2.2 Run tfidf_vec.py to generate tfidf vectors  
2.3 Run get_code_vectors.py, doc2vec.py, get_embedding_weights.py  

3. Compute and weight features  
3.1 track1_calculate_and_weight_features.py compute and weight featurses for the results in the main body of the paper  
3.2 track1_compare_weighting.py builds prediction models with various weigthing  
3.3 track1_compare_weighting_feature_selection.py does predictions with feature selection  
3.4 track1_compare_with_DLKT.py builds models used for comparison with DLKT and IRT  
3.5 track1_organize_data_for_DLKT.py processes data for DLKT. Scripts in dkt as well as sakt and saint build DLKT models  
3.6 irt.Rmd builds the IRT models  
3.7 track1_additional_weighting_comb_calculate_and_weight_features and track1_additional_weighting_comb_compare_weighting.py are for the additional weighting combination presented in the Appendix  

 
