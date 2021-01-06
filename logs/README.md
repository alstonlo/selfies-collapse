## Models

GRU encoder: 

 - Embedding (dim 50)
 - GRU (hidden dim 100, 1 layer) 
 - FC (latent space dim 50)  

GRU decoder: Embedding -> GRU -> FC. Latent vector is feed as the initial 
hidden state of the GRU, and each outputted token is fed as input 
to the GRU for the next time step. 
 



---

## Experiment
