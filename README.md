# Enhance-KG-based-Recommender-System-by-entity-modeling
A PyTorch implementation of the paper "An Extended Knowledge Graph-based Recommendation System Utilizing External Knowledge Base".

# Run the Codes
- Setting 1: Utilizing [MovieLens] entities
'''text
python main_kgat.py --use_pretrain 0 --data_name setting_1 --cf_batch_size 16 --kg_batch_size 32 --test_batch_size 16 --evaluate_every 1
'''
- Setting 2: Utitlizing [MovieLens] entities + is_similar_user + is_similar_item
'''text
python main_kgat.py --use_pretrain 0 --data_name setting_2 --cf_batch_size 16 --kg_batch_size 32 --test_batch_size 16 --evaluate_every 1
'''
- Setting 3: Utitlizing [MovieLens + TMDB] entities + is_similar_cluster + is_similar_storyline
'''text
python main_kgat.py --use_pretrain 0 --data_name setting_3 --cf_batch_size 16 --kg_batch_size 32 --test_batch_size 16 --evaluate_every 1
'''
- Setting 3: Utitlizing [MovieLens + TMDB] entities + is_similar_cluster + is_similar_storyline + is_similar_user + is_similar_item
'''text
python main_kgat.py --use_pretrain 0 --data_name setting_4 --cf_batch_size 16 --kg_batch_size 32 --test_batch_size 16 --evaluate_every 1
'''

# Test the performance of Text embedding features
![image](https://github.com/tree-jhk/Enhance-KG-based-Recommender-System-by-entity-modeling/assets/97151660/fb32f66b-f1fb-45d5-a730-74b4bf183369)

'''text
python main.py
python predict.py
'''
# Obatin KG data
![image](https://github.com/tree-jhk/Enhance-KG-based-Recommender-System-by-entity-modeling/assets/97151660/0e715ad5-e988-4169-8b47-b0e248b913a6)

'''text
# go to RelationModeling folder
run all of the relatiton_modeling.ipynb file
'''
