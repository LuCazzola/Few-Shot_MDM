cd external/motion-diffusion-model

sudo apt update
sudo apt install ffmpeg

# Update environment dependancies
#conda env update --file environment_new.yml
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git

# Body models
mkdir -p body_models
unzip smpl.zip -d body_models/
rm smpl.zip

# Glove dep.
unzip glove.zip
rm glove.zip

#rm -rf kit
unzip t2m.zip
unzip kit.zip
rm t2m.zip
rm kit.zip

cd ../../
echo "Done installing MDM dependancies!"