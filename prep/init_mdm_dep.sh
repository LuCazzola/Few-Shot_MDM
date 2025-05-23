cd external/motion-diffusion-model

sudo apt update
sudo apt install ffmpeg

# Update environment dependancies
conda env update --file environment.yml
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git

bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh

cd ../../
echo "Done installing MDM dependancies!"