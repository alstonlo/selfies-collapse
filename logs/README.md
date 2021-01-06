## VAE Architecture

**Encoder:**

 - Embedding (dim 50)
 - GRU (hidden dim 100, 1 layer) 
 - FC (latent space dim 50)  

**GRU decoder:** Embedding (dim 50) -> GRU -> FC. Latent vector is fed as the
initial hidden state of the GRU, and each outputted token is fed as input 
to the GRU for the next time step. Four sizes of VAE exist (S, M, L, XL), where
the encoder is shared but the decoder size varies: 

 - S: GRU (hidden dim 50, 1 layer)
 - M: GRU (hidden dim 50, 2 layers)
 - L: GRU (hidden dim 50, 3 layers)
 - XL: GRU (hidden dim 100, 3 layers)    

---

## Experiment

All VAEs are trained on QM9 with the same data split, learning rate, 
batch size, and so forth. The main differences between the VAEs are as follows: 

 * Each log directory is of the form `enc={...}_beta={...}`. The `enc={...}`
   species whether the VAE was trained on the SMILES or SELIFES version of QM9.
   The `beta={...}` gives the weight of the KL divergence of the loss used
   to train the VAE (i.e. `loss = recon_loss + beta * KL_divergence`).
 
 * VAEs S, M, and L were trained up to 60 epochs. VAE XL was trained
   without a maximum number of epochs, but until early stopping triggered.  

---

## Viewing Logs

These logs can be viewed using TensorBoard. Once TensorBoard is installed,
and these log files are downloaded, run 

```bash
tensorboard --logdir <log_directory>
```
