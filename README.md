# mds_tf
This is an attempt implement Nonmetric Multidimensional Scaling (MDS) based on Kruskal's 1964 Psychometrika papers (http://cda.psych.uiuc.edu/psychometrika_highly_cited_articles/kruskal_1964a.pdf, http://cda.psych.uiuc.edu/psychometrika_highly_cited_articles/kruskal_1964b.pdf) using TensorFlow in order to: 
1. Demonstrate use of TensorFlow Constraints for monotone regression
2. Use TensorFlow's GradientTape to understand the derivative of the Stress (a.k.a. the MDS loss function) with 
3. Force myself (CL) to:
  a. Read and apply some of Jan De Leeuw's gems (such as https://www.rpubs.com/deleeuw/262795, https://jansweb.netlify.app/publication/deleeuw-e-18-c/, a.o.) to revisit my Quant Psych days
  b. Build up and share the environments needed to run this analysis to build fluency in Docker, poetry and Dynaconf
The first example dataset is a set of pairwise similarities on animals (http://cs.bc.edu/~prudhome/AAPLD/animals.html), which were gathered in a larger study of semantic fluency tasks by Prud’hommeaux and colleagues. 
