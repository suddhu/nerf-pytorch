# wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz
mkdir -p /datasets/sudhars/nerf
cd /datasets/sudhars/nerf
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip -q nerf_example_data.zip
rm nerf_example_data.zip
cd ../../..
