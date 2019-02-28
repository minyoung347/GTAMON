-----------------------------------------------------------
Lablup Just model it 대회 참가 [GTAMON 팀]
-----------------------------------------------------------

제목: 커뮤니티 글을 분석해서 비트코인 가격 변화 예측하기

설명: 

     (1) 비트코인은 내재적 가치가 변하기 보다는 다양한 이슈에 의해서 가격이 변동함
         이슈가 전파되는 Reddit과 같은 커뮤니티의 글을 통해서 비트코인의 가격을 예측하고자 함
         
     (2) Word2vec으로 단어를 임베딩할 때, 커뮤니티 '지식의 변화'를 반영하고자 함
         전체 기간의 말뭉치를 학습하는 것이 아니라, 예측하고자 하는 시점 이전의 자료만 학습하여 단어를 임베딩
         따라서 Online Word2vec을 사용해서 하루 단위로 키워드들의 임베딩이 달라짐
            
     (3) Word2vec으로 임베딩된 벡터로 키워드 네트워크를 구성, 시간에 따른 네트워크 변화 관찰
     
     (4) 네트워크 분석을 통해서 키워드 네트워크의 변화를 측정하고 특징을 뽑음
     
     (5) 네트워크 분석을 통해서 얻은 특징과 인접행렬등의 정보를 RNN 예측모델에 전달
     
     (6) GCN-RNN을 End-to-end로 학습을 시켜서 비트코인의 가격 변화를(up or down) 예측함
     
데이터: 

     Reddit submission title data (2011.1.1 - 2017.12.31), score >= 10
     Bitcoin Daily Price (Bitstamp Exchange, 2011.9.13 - 2017.12.31)

결과: 미완성 

추가로 할 일: 

키워드:  

     Bitcoin, Reddit, Online Word2vec, Keyword Netword, Graph Convolutional Network, Recurrent Neural Network


:whale: Step 1. [1_w2v_training.ipynb]
          
          Training word2vec from titles of raw_reddit_data (2011.1.1 - 2017.12.31)
          
          Input: Reddit jason file compressed by bz2 or xz
          Output: w2v_trained_reddit_{year}_{month}_{day}_{eb_size}_{w2v_win}_{mincount}_{epoch_size}
          
          Hyper_parameters: eb_size: embedding_size of w2v
                            w2v_win: window_size of w2v
                            mincount: mincount of w2v (remove keywords that appear less than mincount)
                            epoch_size: training_epoch of w2v
                            score_threshold: only using data of whose score is larger than score_threshold
           
  
:whale: Step 2. [2_keyword_list.ipynb]
           
           Save keyword_list and keyword_similarity_list from the output of Step.1
           
           Input: w2v_trained_reddit_{year}_{month}_{day}_{eb_size}_{w2v_win}_{mincount}_{epoch_size}
           Output: keyword_{year}_{month}_{day}.csv  - keyword list (label, keyword)
                   keyword_sim_{year}_{month}_{day}.csv   -  keyword_similarity (source, target, similarity)
                   
           Hyper_parameter: sim_threshold = .3 :  using only similarity larger than sim_threshold
           
           
:whale: Step 3. [3_adj_matrix.ipynb]
           
           Convert from similarity_list to adjacency matrix 
           
           Input: keyword_sim_{year}_{month}_{day}.csv
           Output: adj_{year}_{month}_{day}.csv  -  lengh(keyword) by length(keyword) matrix
                                                    1 if connected, 0 otherwise


:whale: Step 4. [4_extract_feature.R]
                                  
           Eextracting 7 features of keyword_network
           1. degree on keyword_network
           2. closeness_centrality on keyword_network
           3. betweenness_centrality on keyword_network
           4. distances from "bitcoin" on keyword_network
           5-7. node binary of egocentric_network (neighborhood 1-3)
                1 if connected, 0 otherwise on egocentric_network
           
           Input:  keyword_{year}_{month}_{day}.csv,  keyword_sim_{year}_{month}_{day}.csv
           Output: features_{year}_{month}_{day}.csv  (# of nodes by 7)
                   from 2013/4/10, because "bitcoin" keyword appears at that day
                   on the conditions eb_size = 50, w2v_win = 5, mincount = 100, epoch_size = 10, score_threshold = 10
                            

:whale: Step 5. [5_gcn_rnn.R]

           Graph Convolutional Neural Network + Recurrent Neural Network (End-to-end training)
           Predict the direction of price change, One-hot(increase, decrease, stop)
           
           Input: features_{year}_{month}_{day}.csv  (# of nodes by 7)
                  adj_{year}_{month}_{day}.csv  -  lengh(keyword) by length(keyword) matrix
                                                    1 if connected, 0 otherwise 
           Output: Loss function, Accuracy of prediction
          
          
:whale: Step 1-1. [1-1_transition_bitcoin.ipynb]
           
           Plot time_series of similarity between the consecutive embedded vectors of 'bitcoin'
           Check the embedded vector of any keyword
           
           
:whale: Step 4-1. [4-1_extract_feature_oneday.R]
                                  
           Same input and output with [extract_feature.R]
           not using multiprocessing
           only extract features on one day

