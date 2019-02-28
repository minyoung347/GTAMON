# extracting features
# 1. degree
# 2. closeness_centrality
# 3. betweenness_centrality
# 4. distances from "bitcoin"
# 5-7. node binary of ego_network (neighborhood 1-3)
# input(1) is keyword_{year}_{month}_{day}.csv
# input(2) is keyword_sim_{year}_{month}_{day}.csv
# output is features_{year}_{month}_{day}.csv  (# of nodes by 7)

if(!require(igraph)) {
  install.packages("igraph")
}
if(!require(parallel)) {
  install.packages("parallel")
}
library(parallel)

setwd('/home/minyoung/data_minyoung/Research/GTAMON_data/share/GTAMON-Lablup/minyoung')

extract_features = function(month){
  library(igraph)
  path_file = "/home/minyoung/data_minyoung/Research/GTAMON_data/keyword/"
  for (year in 2013:2017){
    if (year == 2013 && month <4) {next}
    for (day in 1:31){
      if (year == 2013 && month ==4 && day < 10) {next}
      if (file.exists(sprintf(paste(path_file,"keyword_%d_%d_%d.csv",sep=""),year,month,day))){
        cat(sprintf("in if, %.0f, %.0f, %.0f\n", year, month, day))
        
        node_list = read.csv(file=sprintf(paste(path_file,"keyword_%d_%d_%d.csv",sep=""),year,month,day),header=FALSE, as.is=TRUE)
        edge_list = read.csv(file=sprintf(paste(path_file,"keyword_sim_%d_%d_%d.csv",sep=""),year,month,day),header=FALSE, as.is=TRUE)
        
        colnames(node_list)[1] = "id"
        colnames(node_list)[2] = "keyword"
        colnames(edge_list)[1] = "source"
        colnames(edge_list)[2] = "target"
        colnames(edge_list)[3] = "weight"
        
        # construct network
        net = graph_from_data_frame(d = edge_list, vertices = node_list, directed = FALSE)
        
        # cut the edges below certain threshold
        weight_th = mean(E(net)$weight)
        net_cut = delete_edges(net, E(net)[weight < weight_th])
        
        # ego_network centered at "bitcoin" keyword
        ego_net_1 = make_ego_graph(net_cut, 1, nodes = which(V(net)$keyword=="bitcoin"), mode = c("all"), mindist = 0)[[1]]
        ego_net_2 = make_ego_graph(net_cut, 2, nodes = which(V(net)$keyword=="bitcoin"), mode = c("all"), mindist = 0)[[1]]
        ego_net_3 = make_ego_graph(net_cut, 3, nodes = which(V(net)$keyword=="bitcoin"), mode = c("all"), mindist = 0)[[1]]
        
        # plot network
        # lo = layout_with_fr(ego_net)
        # plot(ego_net,
        #      layout = lo,
        #      vertex.label = V(ego_net)$keyword,
        #      vertex.size = 0,
        #      vertex.shape="none",
        #      edge.arrow.size = .0001,
        #      edge.curved=.3,
        #      main = "bitcoin ego network")

        features = rep(0,nrow(node_list)*7)
        dim(features) = c(nrow(node_list),7)
            
        features[,1] = degree(net) # degree
        features[,2] = round(closeness(net), digits = 6) # closeness_centrality
        features[,3] = betweenness(net) # betweenness_centrality
        features[,4] = round(distances(net, V(net)$keyword == "bitcoin"), digits = 6) # distances from "bitcoin"
        features[as.numeric(as_ids(V(ego_net_1)))+1, 5] = 1 # first-nearest egocentric_network of "bitcoin"
        features[as.numeric(as_ids(V(ego_net_2)))+1, 6] = 1 # second-nearest egocentric_network of "bitcoin"
        features[as.numeric(as_ids(V(ego_net_3)))+1, 7] = 1 # third-nearest egocentric_network of "bitcoin"
        
        write.table(features, file=sprintf(paste(path_file,"features_%d_%d_%d.csv",sep=""),year,month,day), sep=",", row.names=FALSE, col.names=FALSE)
        
        rm(ego_net_1,ego_net_2,ego_net_3,net_cut,weight_th,net,node_list,edge_list,features)
      }
    }
  }
}


#numCores = parallel::detectCores() - 2
myCluster = parallel::makeCluster(4)
clusterExport(myCluster, list("extract_features"))
#parallel::parLapply(cl = myCluster, 1:4, extract_features)
parallel::parLapply(cl = myCluster, 5:8, extract_features)
parallel::parLapply(cl = myCluster, 9:12, extract_features)
stopCluster(myCluster)



