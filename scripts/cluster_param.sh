# !/bin/bash
# ## Nom du job
# SBATCH -J test_openmpi
# ## Choix de la partition (file d'attente dans le vocabulaire SLURM) parmi :
# ## cpu <=> noeud "calcul" pour calcul sur CPU
# ## gpu-v100 <=> noeud avec les deux GPU NVidia V100
# ## gpu-t4 <=> noeud avec les deux GPU NVidia T4
# ## gpu-p5000 <=> noeud avec un GPU NVidia QUADRO P5000
# ## mem <=> noeud avec beaucoup de memoire RAM
# SBATCH -p cpu
# ## Choix du "groupe"
# SBATCH -A lamia
# ## Choix du nombre de noeuds
# SBATCH -N 1
# ## Choix du nombre de processus
# SBATCH -n 900
# ## Envoi d'un courriel a la fin du job
# SBATCH --mail-type END
# SBATCH --mail-user theofigini@gmail.com
# ## lancement du programme
# mpirun -np 1 ./master_mpi : -np 899 ./slave_mpi