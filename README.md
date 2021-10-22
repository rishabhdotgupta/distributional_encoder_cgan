# distributional_encoder_cgan

For parametric data, run:  
```python cgan_parametric.py --batch_size 32 --set_size 500 --n_epochs 50 --enc_dim 2 --latent_dim 20 --num_paths 100 --reg 0```

For point clouds, run:  
```python cgan_pc.py --batch_size 16 --set_size 100 --n_epochs 50 --enc_dim 256 --latent_dim 100 --num_paths 50 --reg 0```

For sciplex data, run:
```python setenc_cwgan_gp_sciplex.py --n_epochs 100 --sample_interval 1000 --batch_size 32 --latent_dim 100 --n_classes 32 --enc_dim 10```

